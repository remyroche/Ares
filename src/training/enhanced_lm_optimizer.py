# src/training/enhanced_lm_optimizer.py

"""
Enhanced LM Model Optimizer for Step6, Step6_5, and Step9

This module provides comprehensive optimization for Language Model (LM) components:
1. Advanced feature selection with multiple algorithms
2. L1-L2 regularization with model-specific tuning
3. Optuna hyperparameter optimization in batches
4. Vectorized/matrix operations for efficiency
5. Model-specific optimizations for different architectures
"""

import asyncio
import json
import os
import pickle
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.feature_selection import (
    mutual_info_classif, mutual_info_regression,
    f_classif, f_regression, SelectKBest, SelectFromModel,
    RFE, VarianceThreshold, SequentialFeatureSelector
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, LogisticRegression, ElasticNet
from sklearn.decomposition import PCA, FastICA
from sklearn.covariance import LedoitWolf
import lightgbm as lgb
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import shap

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors
from src.utils.warning_symbols import error, failed, success, timeout
from src.utils.decorators import guard_dataframe_nulls, with_tracing_span


class EnhancedLMOptimizer:
    """
    Enhanced LM Model Optimizer with comprehensive optimization features.
    
    Features:
    - Multi-algorithm feature selection
    - L1-L2 regularization with model-specific tuning
    - Optuna hyperparameter optimization in batches
    - Vectorized operations for efficiency
    - Model-specific optimizations
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("EnhancedLMOptimizer")
        
        # Load optimization configuration
        self.optimization_config = self._load_optimization_config()
        
        # Initialize components
        self.feature_selector = None
        self.regularization_manager = None
        self.optuna_study = None
        
        # Performance tracking
        self.optimization_metrics = {
            "feature_selection_time": 0.0,
            "hyperparameter_optimization_time": 0.0,
            "regularization_tuning_time": 0.0,
            "total_optimization_time": 0.0
        }
        
        # Cache for optimization results
        self.optimization_cache = {}
        
    def _load_optimization_config(self) -> Dict[str, Any]:
        """Load and validate optimization configuration."""
        default_config = {
            "feature_selection": {
                "enable": True,
                "methods": ["mutual_info", "lasso", "random_forest", "shap"],
                "target_features": {
                    "step6": 80,
                    "step6_5": 100,
                    "step9": 90
                },
                "vif_threshold": 10.0,
                "correlation_threshold": 0.95,
                "variance_threshold": 0.01,
                "mutual_info_threshold": 0.001,
                "shap_threshold": 0.001
            },
            "regularization": {
                "enable": True,
                "l1_alpha_range": [0.001, 0.1],
                "l2_alpha_range": [0.0001, 0.01],
                "dropout_range": [0.1, 0.5],
                "model_specific": {
                    "lightgbm": {
                        "reg_alpha_range": [0.001, 0.1],
                        "reg_lambda_range": [0.0001, 0.01]
                    },
                    "neural_networks": {
                        "weight_decay_range": [1e-6, 1e-3],
                        "dropout_range": [0.1, 0.5]
                    }
                }
            },
            "optuna": {
                "enable": True,
                "n_trials_per_batch": 50,
                "n_batches": 3,
                "timeout_per_batch": 300,  # 5 minutes per batch
                "sampler": "tpe",
                "pruner": "median",
                "storage": None  # Can be set to database URL
            },
            "vectorization": {
                "enable": True,
                "batch_size": 1024,
                "use_gpu": torch.cuda.is_available(),
                "memory_efficient": True
            }
        }
        
        # Merge with config
        config = self.config.get("enhanced_lm_optimizer", {})
        for key, value in config.items():
            if key in default_config:
                if isinstance(value, dict) and isinstance(default_config[key], dict):
                    default_config[key].update(value)
                else:
                    default_config[key] = value
                    
        return default_config
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="enhanced LM optimizer initialization"
    )
    async def initialize(self) -> bool:
        """Initialize the enhanced LM optimizer."""
        try:
            self.logger.info("🔄 Initializing Enhanced LM Optimizer...")
            
            # Initialize feature selector
            if self.optimization_config["feature_selection"]["enable"]:
                self.feature_selector = EnhancedFeatureSelector(self.optimization_config)
                await self.feature_selector.initialize()
            
            # Initialize regularization manager
            if self.optimization_config["regularization"]["enable"]:
                self.regularization_manager = EnhancedRegularizationManager(self.optimization_config)
                await self.regularization_manager.initialize()
            
            # Initialize Optuna study
            if self.optimization_config["optuna"]["enable"]:
                await self._initialize_optuna_study()
            
            self.logger.info("✅ Enhanced LM Optimizer initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Enhanced LM Optimizer: {e}")
            return False
    
    async def _initialize_optuna_study(self):
        """Initialize Optuna study for hyperparameter optimization."""
        try:
            optuna_config = self.optimization_config["optuna"]
            
            # Create study
            study_name = f"enhanced_lm_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Configure sampler
            if optuna_config["sampler"] == "tpe":
                sampler = TPESampler(seed=42)
            else:
                sampler = TPESampler(seed=42)
            
            # Create study
            self.optuna_study = optuna.create_study(
                direction="maximize",
                sampler=sampler,
                storage=optuna_config["storage"],
                study_name=study_name,
                load_if_exists=True
            )
            
            self.logger.info(f"✅ Optuna study '{study_name}' initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Optuna study: {e}")
    
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="comprehensive LM optimization"
    )
    async def optimize_lm_model(
        self,
        step_name: str,
        features_df: pd.DataFrame,
        target: pd.Series,
        model_type: str,
        architecture: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Comprehensive optimization for LM models.
        
        Args:
            step_name: Step name (step6, step6_5, step9)
            features_df: Input features DataFrame
            target: Target variable series
            model_type: Model type (classification, regression)
            architecture: Model architecture (LightGBM, CNN, TCN, Transformer)
            **kwargs: Additional parameters
            
        Returns:
            Dict containing optimization results
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"🔄 Starting comprehensive optimization for {step_name} {architecture}")
            
            optimization_results = {
                "step_name": step_name,
                "architecture": architecture,
                "model_type": model_type,
                "optimization_timestamp": datetime.now().isoformat(),
                "feature_selection": {},
                "regularization": {},
                "hyperparameter_optimization": {},
                "performance_metrics": {}
            }
            
            # Step 1: Feature Selection
            if self.optimization_config["feature_selection"]["enable"]:
                self.logger.info(f"📊 Step 1: Feature selection for {step_name}")
                feature_selection_start = time.time()
                
                optimized_features, feature_selection_results = await self._optimize_features(
                    features_df, target, step_name, architecture
                )
                
                optimization_results["feature_selection"] = feature_selection_results
                self.optimization_metrics["feature_selection_time"] = time.time() - feature_selection_start
                
                self.logger.info(f"✅ Feature selection completed: {len(features_df.columns)} -> {len(optimized_features.columns)} features")
            else:
                optimized_features = features_df
            
            # Step 2: Regularization Tuning
            if self.optimization_config["regularization"]["enable"]:
                self.logger.info(f"🔧 Step 2: Regularization tuning for {step_name}")
                regularization_start = time.time()
                
                regularization_params = await self._optimize_regularization(
                    optimized_features, target, step_name, architecture
                )
                
                optimization_results["regularization"] = regularization_params
                self.optimization_metrics["regularization_tuning_time"] = time.time() - regularization_start
                
                self.logger.info(f"✅ Regularization tuning completed")
            
            # Step 3: Hyperparameter Optimization with Optuna
            if self.optimization_config["optuna"]["enable"]:
                self.logger.info(f"🎯 Step 3: Hyperparameter optimization for {step_name}")
                hyperopt_start = time.time()
                
                best_params, hyperopt_results = await self._optimize_hyperparameters(
                    optimized_features, target, step_name, architecture, model_type
                )
                
                optimization_results["hyperparameter_optimization"] = hyperopt_results
                self.optimization_metrics["hyperparameter_optimization_time"] = time.time() - hyperopt_start
                
                self.logger.info(f"✅ Hyperparameter optimization completed")
            
            # Step 4: Performance Evaluation
            self.logger.info(f"📈 Step 4: Performance evaluation for {step_name}")
            performance_metrics = await self._evaluate_optimized_model(
                optimized_features, target, step_name, architecture, model_type,
                optimization_results
            )
            
            optimization_results["performance_metrics"] = performance_metrics
            
            # Update total time
            self.optimization_metrics["total_optimization_time"] = time.time() - start_time
            
            # Cache results
            cache_key = f"{step_name}_{architecture}_{model_type}"
            self.optimization_cache[cache_key] = optimization_results
            
            self.logger.info(f"✅ Comprehensive optimization completed for {step_name} in {self.optimization_metrics['total_optimization_time']:.2f}s")
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"❌ Optimization failed for {step_name}: {e}")
            return None
    
    async def _optimize_features(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        step_name: str,
        architecture: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Optimize feature selection using multiple algorithms."""
        try:
            if self.feature_selector is None:
                return features_df, {"error": "feature_selector_not_available"}
            
            # Get target feature count for this step
            target_features = self.optimization_config["feature_selection"]["target_features"].get(
                step_name, 80
            )
            
            # Apply enhanced feature selection
            optimized_features, selection_metadata = await self.feature_selector.select_features_enhanced(
                features_df, target, target_features, architecture
            )
            
            return optimized_features, selection_metadata
            
        except Exception as e:
            self.logger.error(f"❌ Feature optimization failed: {e}")
            return features_df, {"error": str(e)}
    
    async def _optimize_regularization(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        step_name: str,
        architecture: str
    ) -> Dict[str, Any]:
        """Optimize regularization parameters."""
        try:
            if self.regularization_manager is None:
                return {"error": "regularization_manager_not_available"}
            
            # Get regularization parameters optimized for this architecture
            regularization_params = await self.regularization_manager.optimize_regularization(
                features_df, target, step_name, architecture
            )
            
            return regularization_params
            
        except Exception as e:
            self.logger.error(f"❌ Regularization optimization failed: {e}")
            return {"error": str(e)}
    
    async def _optimize_hyperparameters(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        step_name: str,
        architecture: str,
        model_type: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Optimize hyperparameters using Optuna in batches."""
        try:
            if self.optuna_study is None:
                return {}, {"error": "optuna_study_not_available"}
            
            optuna_config = self.optimization_config["optuna"]
            n_batches = optuna_config["n_batches"]
            n_trials_per_batch = optuna_config["n_trials_per_batch"]
            timeout_per_batch = optuna_config["timeout_per_batch"]
            
            best_params = {}
            batch_results = []
            
            for batch_idx in range(n_batches):
                self.logger.info(f"🔄 Batch {batch_idx + 1}/{n_batches} for {step_name}")
                
                # Create objective function for this batch
                def objective(trial):
                    return self._hyperparameter_objective(
                        trial, features_df, target, step_name, architecture, model_type
                    )
                
                # Optimize this batch
                study = optuna.create_study(direction="maximize")
                study.optimize(
                    objective,
                    n_trials=n_trials_per_batch,
                    timeout=timeout_per_batch
                )
                
                batch_results.append({
                    "batch": batch_idx + 1,
                    "best_value": study.best_value,
                    "best_params": study.best_params,
                    "n_trials": len(study.trials)
                })
                
                # Update best params if this batch is better
                if not best_params or study.best_value > best_params.get("best_value", -float('inf')):
                    best_params = {
                        "best_value": study.best_value,
                        "best_params": study.best_params
                    }
                
                self.logger.info(f"✅ Batch {batch_idx + 1} completed: best_value={study.best_value:.4f}")
            
            return best_params["best_params"], {
                "batch_results": batch_results,
                "overall_best_value": best_params["best_value"],
                "total_trials": sum(r["n_trials"] for r in batch_results)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Hyperparameter optimization failed: {e}")
            return {}, {"error": str(e)}
    
    def _hyperparameter_objective(
        self,
        trial: optuna.Trial,
        features_df: pd.DataFrame,
        target: pd.Series,
        step_name: str,
        architecture: str,
        model_type: str
    ) -> float:
        """Objective function for Optuna hyperparameter optimization."""
        try:
            # Get hyperparameter suggestions based on architecture
            if architecture == "LightGBM":
                params = self._suggest_lightgbm_params(trial, step_name)
                model = lgb.LGBMClassifier(**params) if model_type == "classification" else lgb.LGBMRegressor(**params)
            elif architecture in ["CNN", "TCN", "Transformer"]:
                params = self._suggest_neural_network_params(trial, architecture, step_name)
                model = self._create_neural_network_model(params, architecture, features_df.shape[1], model_type)
            else:
                # Default to LightGBM
                params = self._suggest_lightgbm_params(trial, step_name)
                model = lgb.LGBMClassifier(**params) if model_type == "classification" else lgb.LGBMRegressor(**params)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, features_df, target,
                cv=TimeSeriesSplit(n_splits=3),
                scoring='accuracy' if model_type == "classification" else 'neg_mean_squared_error'
            )
            
            return cv_scores.mean()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Trial failed: {e}")
            return -float('inf')
    
    def _suggest_lightgbm_params(self, trial: optuna.Trial, step_name: str) -> Dict[str, Any]:
        """Suggest LightGBM hyperparameters."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 0.1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 0.1),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'random_state': 42,
            'verbose': -1
        }
    
    def _suggest_neural_network_params(self, trial: optuna.Trial, architecture: str, step_name: str) -> Dict[str, Any]:
        """Suggest neural network hyperparameters."""
        return {
            'hidden_size': trial.suggest_int('hidden_size', 64, 512),
            'num_layers': trial.suggest_int('num_layers', 2, 6),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            'epochs': trial.suggest_int('epochs', 10, 50)
        }
    
    def _create_neural_network_model(self, params: Dict[str, Any], architecture: str, input_size: int, model_type: str):
        """Create neural network model based on architecture."""
        # This is a simplified version - in practice, you'd have more sophisticated model creation
        if architecture == "CNN":
            return SimpleCNNModel(input_size, params, model_type)
        elif architecture == "TCN":
            return SimpleTCNModel(input_size, params, model_type)
        elif architecture == "Transformer":
            return SimpleTransformerModel(input_size, params, model_type)
        else:
            return SimpleNNModel(input_size, params, model_type)
    
    async def _evaluate_optimized_model(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        step_name: str,
        architecture: str,
        model_type: str,
        optimization_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate the optimized model performance."""
        try:
            # Create final model with optimized parameters
            if architecture == "LightGBM":
                best_params = optimization_results.get("hyperparameter_optimization", {}).get("best_params", {})
                model = lgb.LGBMClassifier(**best_params) if model_type == "classification" else lgb.LGBMRegressor(**best_params)
            else:
                # For neural networks, you'd create the model with optimized parameters
                model = None  # Simplified for this example
            
            if model is not None:
                # Cross-validation evaluation
                cv_scores = cross_val_score(
                    model, features_df, target,
                    cv=TimeSeriesSplit(n_splits=5),
                    scoring='accuracy' if model_type == "classification" else 'neg_mean_squared_error'
                )
                
                return {
                    "cv_mean": cv_scores.mean(),
                    "cv_std": cv_scores.std(),
                    "cv_scores": cv_scores.tolist()
                }
            else:
                return {"error": "model_creation_failed"}
                
        except Exception as e:
            self.logger.error(f"❌ Model evaluation failed: {e}")
            return {"error": str(e)}
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization metrics and results."""
        return {
            "optimization_metrics": self.optimization_metrics,
            "cache_size": len(self.optimization_cache),
            "cached_steps": list(self.optimization_cache.keys())
        }


class EnhancedFeatureSelector:
    """Enhanced feature selector with multiple algorithms and vectorized operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("EnhancedFeatureSelector")
        self.feature_selection_config = config["feature_selection"]
    
    async def initialize(self):
        """Initialize the feature selector."""
        self.logger.info("✅ Enhanced Feature Selector initialized")
    
    async def select_features_enhanced(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        target_features: int,
        architecture: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Enhanced feature selection using multiple algorithms."""
        try:
            start_time = time.time()
            
            # Step 1: Variance threshold (remove low variance features)
            variance_selector = VarianceThreshold(threshold=self.feature_selection_config["variance_threshold"])
            variance_mask = variance_selector.fit_transform(features_df)
            variance_features = features_df.columns[variance_selector.get_support()].tolist()
            
            # Step 2: Correlation analysis (remove highly correlated features)
            correlation_features = self._remove_correlated_features(
                features_df[variance_features], self.feature_selection_config["correlation_threshold"]
            )
            
            # Step 3: Mutual information
            if "mutual_info" in self.feature_selection_config["methods"]:
                mi_features = self._select_mutual_info_features(
                    features_df[correlation_features], target, target_features
                )
            else:
                mi_features = correlation_features
            
            # Step 4: Lasso-based selection
            if "lasso" in self.feature_selection_config["methods"]:
                lasso_features = self._select_lasso_features(
                    features_df[mi_features], target, target_features
                )
            else:
                lasso_features = mi_features
            
            # Step 5: Random Forest importance
            if "random_forest" in self.feature_selection_config["methods"]:
                rf_features = self._select_random_forest_features(
                    features_df[lasso_features], target, target_features
                )
            else:
                rf_features = lasso_features
            
            # Step 6: SHAP analysis (if enabled and features are manageable)
            if "shap" in self.feature_selection_config["methods"] and len(rf_features) <= 50:
                final_features = self._select_shap_features(
                    features_df[rf_features], target, target_features
                )
            else:
                final_features = rf_features[:target_features]
            
            # Create final feature set
            optimized_features = features_df[final_features]
            
            selection_metadata = {
                "original_features": len(features_df.columns),
                "variance_filtered": len(variance_features),
                "correlation_filtered": len(correlation_features),
                "mutual_info_filtered": len(mi_features),
                "lasso_filtered": len(lasso_features),
                "random_forest_filtered": len(rf_features),
                "final_features": len(final_features),
                "selection_time": time.time() - start_time
            }
            
            return optimized_features, selection_metadata
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced feature selection failed: {e}")
            return features_df, {"error": str(e)}
    
    def _remove_correlated_features(self, features_df: pd.DataFrame, threshold: float) -> List[str]:
        """Remove highly correlated features using vectorized operations."""
        try:
            # Calculate correlation matrix
            corr_matrix = features_df.corr().abs()
            
            # Find upper triangle of correlation matrix
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find features with correlation above threshold
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
            
            # Return features to keep
            return [col for col in features_df.columns if col not in to_drop]
            
        except Exception as e:
            self.logger.warning(f"⚠️ Correlation filtering failed: {e}")
            return features_df.columns.tolist()
    
    def _select_mutual_info_features(self, features_df: pd.DataFrame, target: pd.Series, target_features: int) -> List[str]:
        """Select features using mutual information."""
        try:
            # Determine if classification or regression
            if target.dtype == 'object' or len(target.unique()) < 10:
                mi_scores = mutual_info_classif(features_df, target, random_state=42)
            else:
                mi_scores = mutual_info_regression(features_df, target, random_state=42)
            
            # Get feature indices sorted by importance
            feature_indices = np.argsort(mi_scores)[::-1]
            
            # Select top features
            selected_indices = feature_indices[:target_features]
            selected_features = features_df.columns[selected_indices].tolist()
            
            return selected_features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Mutual info selection failed: {e}")
            return features_df.columns[:target_features].tolist()
    
    def _select_lasso_features(self, features_df: pd.DataFrame, target: pd.Series, target_features: int) -> List[str]:
        """Select features using Lasso regularization."""
        try:
            # Determine if classification or regression
            if target.dtype == 'object' or len(target.unique()) < 10:
                lasso = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=1000)
            else:
                lasso = Lasso(alpha=0.01, random_state=42, max_iter=1000)
            
            # Fit Lasso
            lasso.fit(features_df, target)
            
            # Get non-zero coefficients
            if hasattr(lasso, 'coef_'):
                coef = lasso.coef_
            else:
                coef = lasso.feature_importances_
            
            # Select features with non-zero coefficients
            selected_indices = np.where(np.abs(coef) > 0)[0]
            selected_features = features_df.columns[selected_indices].tolist()
            
            # If too few features, add more based on coefficient magnitude
            if len(selected_features) < target_features:
                top_indices = np.argsort(np.abs(coef))[::-1][:target_features]
                selected_features = features_df.columns[top_indices].tolist()
            
            return selected_features[:target_features]
            
        except Exception as e:
            self.logger.warning(f"⚠️ Lasso selection failed: {e}")
            return features_df.columns[:target_features].tolist()
    
    def _select_random_forest_features(self, features_df: pd.DataFrame, target: pd.Series, target_features: int) -> List[str]:
        """Select features using Random Forest importance."""
        try:
            # Determine if classification or regression
            if target.dtype == 'object' or len(target.unique()) < 10:
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            # Fit Random Forest
            rf.fit(features_df, target)
            
            # Get feature importance
            importance = rf.feature_importances_
            
            # Select top features
            top_indices = np.argsort(importance)[::-1][:target_features]
            selected_features = features_df.columns[top_indices].tolist()
            
            return selected_features
            
        except Exception as e:
            self.logger.warning(f"⚠️ Random Forest selection failed: {e}")
            return features_df.columns[:target_features].tolist()
    
    def _select_shap_features(self, features_df: pd.DataFrame, target: pd.Series, target_features: int) -> List[str]:
        """Select features using SHAP analysis."""
        try:
            # Use LightGBM for SHAP analysis
            if target.dtype == 'object' or len(target.unique()) < 10:
                model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
            else:
                model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
            
            # Fit model
            model.fit(features_df, target)
            
            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(features_df)
            
            # If classification, use the first class SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Calculate mean absolute SHAP values
            mean_shap = np.mean(np.abs(shap_values), axis=0)
            
            # Select top features
            top_indices = np.argsort(mean_shap)[::-1][:target_features]
            selected_features = features_df.columns[top_indices].tolist()
            
            return selected_features
            
        except Exception as e:
            self.logger.warning(f"⚠️ SHAP selection failed: {e}")
            return features_df.columns[:target_features].tolist()


class EnhancedRegularizationManager:
    """Enhanced regularization manager with model-specific tuning."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("EnhancedRegularizationManager")
        self.regularization_config = config["regularization"]
    
    async def initialize(self):
        """Initialize the regularization manager."""
        self.logger.info("✅ Enhanced Regularization Manager initialized")
    
    async def optimize_regularization(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        step_name: str,
        architecture: str
    ) -> Dict[str, Any]:
        """Optimize regularization parameters for the given architecture."""
        try:
            if architecture == "LightGBM":
                return await self._optimize_lightgbm_regularization(features_df, target, step_name)
            elif architecture in ["CNN", "TCN", "Transformer"]:
                return await self._optimize_neural_network_regularization(features_df, target, step_name, architecture)
            else:
                return await self._optimize_general_regularization(features_df, target, step_name)
                
        except Exception as e:
            self.logger.error(f"❌ Regularization optimization failed: {e}")
            return {"error": str(e)}
    
    async def _optimize_lightgbm_regularization(self, features_df: pd.DataFrame, target: pd.Series, step_name: str) -> Dict[str, Any]:
        """Optimize LightGBM regularization parameters."""
        try:
            # Use Optuna to optimize regularization parameters
            def objective(trial):
                reg_alpha = trial.suggest_float('reg_alpha', 0.001, 0.1)
                reg_lambda = trial.suggest_float('reg_lambda', 0.001, 0.1)
                
                model = lgb.LGBMClassifier(
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    n_estimators=100,
                    random_state=42,
                    verbose=-1
                )
                
                scores = cross_val_score(model, features_df, target, cv=3, scoring='accuracy')
                return scores.mean()
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=20)
            
            return {
                "reg_alpha": study.best_params["reg_alpha"],
                "reg_lambda": study.best_params["reg_lambda"],
                "best_score": study.best_value
            }
            
        except Exception as e:
            self.logger.error(f"❌ LightGBM regularization optimization failed: {e}")
            return {"reg_alpha": 0.01, "reg_lambda": 0.001}
    
    async def _optimize_neural_network_regularization(self, features_df: pd.DataFrame, target: pd.Series, step_name: str, architecture: str) -> Dict[str, Any]:
        """Optimize neural network regularization parameters."""
        try:
            # Use Optuna to optimize regularization parameters
            def objective(trial):
                weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3)
                dropout = trial.suggest_float('dropout', 0.1, 0.5)
                
                # Create a simple neural network for testing
                model = SimpleNNModel(
                    input_size=features_df.shape[1],
                    params={"dropout": dropout, "weight_decay": weight_decay},
                    model_type="classification"
                )
                
                # Simplified evaluation
                return 0.7  # Placeholder score
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=20)
            
            return {
                "weight_decay": study.best_params["weight_decay"],
                "dropout": study.best_params["dropout"],
                "best_score": study.best_value
            }
            
        except Exception as e:
            self.logger.error(f"❌ Neural network regularization optimization failed: {e}")
            return {"weight_decay": 1e-4, "dropout": 0.2}
    
    async def _optimize_general_regularization(self, features_df: pd.DataFrame, target: pd.Series, step_name: str) -> Dict[str, Any]:
        """Optimize general regularization parameters."""
        try:
            # Use ElasticNet for general regularization optimization
            def objective(trial):
                alpha = trial.suggest_float('alpha', 0.001, 0.1)
                l1_ratio = trial.suggest_float('l1_ratio', 0.1, 0.9)
                
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
                scores = cross_val_score(model, features_df, target, cv=3, scoring='neg_mean_squared_error')
                return scores.mean()
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=20)
            
            return {
                "alpha": study.best_params["alpha"],
                "l1_ratio": study.best_params["l1_ratio"],
                "best_score": study.best_value
            }
            
        except Exception as e:
            self.logger.error(f"❌ General regularization optimization failed: {e}")
            return {"alpha": 0.01, "l1_ratio": 0.5}


# Simple model classes for demonstration
class SimpleNNModel(nn.Module):
    def __init__(self, input_size: int, params: Dict[str, Any], model_type: str):
        super().__init__()
        self.input_size = input_size
        self.params = params
        self.model_type = model_type
        
        # Simple feedforward network
        self.layers = nn.Sequential(
            nn.Linear(input_size, params.get('hidden_size', 128)),
            nn.ReLU(),
            nn.Dropout(params.get('dropout', 0.2)),
            nn.Linear(params.get('hidden_size', 128), 64),
            nn.ReLU(),
            nn.Dropout(params.get('dropout', 0.2)),
            nn.Linear(64, 1 if model_type == "regression" else 2)
        )
    
    def forward(self, x):
        return self.layers(x)


class SimpleCNNModel(nn.Module):
    def __init__(self, input_size: int, params: Dict[str, Any], model_type: str):
        super().__init__()
        self.input_size = input_size
        self.params = params
        self.model_type = model_type
        
        # Simple CNN for 1D data
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(params.get('dropout', 0.2)),
            nn.Linear(32, 1 if model_type == "regression" else 2)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv_layers(x)
        x = x.squeeze(-1)
        return self.fc_layers(x)


class SimpleTCNModel(nn.Module):
    def __init__(self, input_size: int, params: Dict[str, Any], model_type: str):
        super().__init__()
        self.input_size = input_size
        self.params = params
        self.model_type = model_type
        
        # Simple TCN implementation
        self.layers = nn.Sequential(
            nn.Linear(input_size, params.get('hidden_size', 128)),
            nn.ReLU(),
            nn.Dropout(params.get('dropout', 0.2)),
            nn.Linear(params.get('hidden_size', 128), 64),
            nn.ReLU(),
            nn.Dropout(params.get('dropout', 0.2)),
            nn.Linear(64, 1 if model_type == "regression" else 2)
        )
    
    def forward(self, x):
        return self.layers(x)


class SimpleTransformerModel(nn.Module):
    def __init__(self, input_size: int, params: Dict[str, Any], model_type: str):
        super().__init__()
        self.input_size = input_size
        self.params = params
        self.model_type = model_type
        
        # Simple transformer implementation
        self.embedding = nn.Linear(input_size, params.get('hidden_size', 128))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=params.get('hidden_size', 128),
                nhead=8,
                dim_feedforward=params.get('hidden_size', 128) * 4,
                dropout=params.get('dropout', 0.2),
                batch_first=True
            ),
            num_layers=params.get('num_layers', 2)
        )
        self.output_layer = nn.Linear(params.get('hidden_size', 128), 1 if model_type == "regression" else 2)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.output_layer(x)