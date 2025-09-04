#!/usr/bin/env python3
"""
Enhanced Parameter Optimization Step

Enhanced version of parameter optimization with comprehensive protection:
- Data validation and integrity checks
- Error handling and recovery
- Performance monitoring
- State management
- Advanced optimization algorithms
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import pickle

from src.utils.logger import system_logger
from src.training.steps.optimisation.optimisation_decorators import (
    protect_optimisation_operation,
    protect_data_operation,
    data_protection,
    error_handling,
    performance_monitoring,
    operation_logging
)
from src.training.steps.optimisation.optimisation_utilities import (
    get_data_formatting_utils,
    get_analysis_operations_utils,
    get_data_access_control,
    get_pipeline_state_manager,
    get_performance_optimizer
)
from src.utils.pipeline_protection_framework import (
    ValidationLevel,
    OperationType,
    DataIntegrityCheck
)
from src.utils.common_operations import (
    ensure_directory,
    safe_file_exists,
    safe_json_dump,
    safe_json_load,
    format_datetime,
    get_current_datetime
)


class EnhancedParameterOptimizationStep:
    """Enhanced parameter optimization step with comprehensive protection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("EnhancedParameterOptimizationStep")
        
        # Initialize utilities
        self.data_formatter = get_data_formatting_utils()
        self.analysis_ops = get_analysis_operations_utils()
        self.data_access = get_data_access_control()
        self.state_manager = get_pipeline_state_manager()
        self.performance_optimizer = get_performance_optimizer()
        
        # Configuration
        self.optimization_methods = config.get("optimization_methods", ["grid_search", "random_search", "bayesian"])
        self.cv_folds = config.get("cv_folds", 5)
        self.random_state = config.get("random_state", 42)
        self.n_trials = config.get("n_trials", 100)
        self.timeout_seconds = config.get("timeout_seconds", 3600)  # 1 hour
        self.min_improvement = config.get("min_improvement", 0.01)  # 1% improvement
        
    @protect_optimisation_operation(ValidationLevel.CRITICAL)
    @data_protection(ValidationLevel.CRITICAL, backup_enabled=True)
    @error_handling(retry_count=2, critical_errors=["data_corruption", "insufficient_data"])
    @performance_monitoring(alert_threshold=1800.0)  # 30 minutes
    @operation_logging(log_level="INFO", audit_trail=True)
    async def optimize_parameters(self, 
                                symbol: str,
                                exchange: str,
                                timeframe: str,
                                data_dir: str) -> bool:
        """Optimize parameters with comprehensive protection."""
        try:
            self.logger.info(f"🔧 Starting enhanced parameter optimization for {symbol} on {exchange}")
            
            # Step 1: Load and validate data
            training_data = await self._load_and_validate_training_data(symbol, exchange, data_dir)
            if training_data is None:
                return False
            
            # Step 2: Load models
            models = await self._load_models(symbol, exchange, data_dir)
            if models is None:
                return False
            
            # Step 3: Define parameter spaces
            parameter_spaces = await self._define_parameter_spaces(models)
            if parameter_spaces is None:
                return False
            
            # Step 4: Perform optimization
            optimization_results = await self._perform_optimization(training_data, models, parameter_spaces)
            if optimization_results is None:
                return False
            
            # Step 5: Validate optimization results
            validation_result = await self._validate_optimization_results(optimization_results)
            if not validation_result:
                return False
            
            # Step 6: Save results
            success = await self._save_optimization_results(optimization_results, symbol, exchange, data_dir)
            
            if success:
                self.logger.info("✅ Enhanced parameter optimization completed successfully")
                return True
            else:
                self.logger.error("❌ Failed to save optimization results")
                return False
                
        except Exception as e:
            self.logger.exception(f"❌ Enhanced parameter optimization failed: {e}")
            return False
    
    @protect_data_operation(ValidationLevel.STANDARD)
    async def _load_and_validate_training_data(self, 
                                             symbol: str,
                                             exchange: str,
                                             data_dir: str) -> Optional[Dict[str, Any]]:
        """Load and validate training data."""
        try:
            self.logger.info("📁 Loading and validating training data...")
            
            # Load feature data
            feature_file = f"{data_dir}/{exchange}_{symbol}_feature_engineered_data.pkl"
            features = self.data_access.secure_data_loading(
                feature_file,
                user_id="parameter_optimization",
                validate_integrity=True
            )
            
            if features is None:
                self.logger.error(f"❌ Failed to load features from {feature_file}")
                return None
            
            # Load target data
            target_file = f"{data_dir}/{exchange}_{symbol}_target_data.pkl"
            targets = self.data_access.secure_data_loading(
                target_file,
                user_id="parameter_optimization",
                validate_integrity=True
            )
            
            if targets is None:
                self.logger.error(f"❌ Failed to load targets from {target_file}")
                return None
            
            # Validate data structure
            if not isinstance(features, (pd.DataFrame, np.ndarray)):
                self.logger.error("❌ Features must be DataFrame or numpy array")
                return None
            
            if not isinstance(targets, (pd.Series, np.ndarray)):
                self.logger.error("❌ Targets must be Series or numpy array")
                return None
            
            # Convert to consistent format
            if isinstance(features, pd.DataFrame):
                X = features.values
                feature_names = features.columns.tolist()
            else:
                X = features
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
            if isinstance(targets, pd.Series):
                y = targets.values
            else:
                y = targets
            
            # Validate dimensions
            if len(X) != len(y):
                self.logger.error(f"❌ Feature and target dimensions mismatch: {len(X)} vs {len(y)}")
                return None
            
            # Check for sufficient samples
            if len(X) < 100:
                self.logger.error(f"❌ Insufficient samples for optimization: {len(X)}")
                return None
            
            # Handle missing values
            if isinstance(X, np.ndarray):
                if np.isnan(X).any():
                    self.logger.warning("⚠️ Missing values detected in features, filling with median")
                    from sklearn.impute import SimpleImputer
                    imputer = SimpleImputer(strategy='median')
                    X = imputer.fit_transform(X)
            
            training_data = {
                "X": X,
                "y": y,
                "feature_names": feature_names,
                "n_samples": len(X),
                "n_features": X.shape[1] if X.ndim > 1 else 1,
                "target_distribution": np.bincount(y.astype(int)) if y.dtype in [np.int32, np.int64] else None
            }
            
            self.logger.info(f"✅ Training data loaded and validated: {training_data['n_samples']} samples, {training_data['n_features']} features")
            return training_data
            
        except Exception as e:
            self.logger.exception(f"❌ Training data loading failed: {e}")
            return None
    
    @protect_data_operation(ValidationLevel.STANDARD)
    async def _load_models(self, 
                         symbol: str,
                         exchange: str,
                         data_dir: str) -> Optional[Dict[str, Any]]:
        """Load models for optimization."""
        try:
            self.logger.info("🤖 Loading models for optimization...")
            
            # Load trained models
            models_file = f"{data_dir}/{exchange}_{symbol}_trained_models.pkl"
            models = self.data_access.secure_data_loading(
                models_file,
                user_id="parameter_optimization",
                validate_integrity=False  # Custom validation for models
            )
            
            if models is None:
                self.logger.error(f"❌ Failed to load models from {models_file}")
                return None
            
            # Validate models structure
            if not isinstance(models, dict):
                self.logger.error("❌ Models must be a dictionary")
                return None
            
            # Check for required model types
            required_model_types = ["classifier", "regressor", "ensemble"]
            available_model_types = list(models.keys())
            
            if not any(model_type in available_model_types for model_type in required_model_types):
                self.logger.error(f"❌ No required model types found. Available: {available_model_types}")
                return None
            
            # Validate each model
            validated_models = {}
            for model_type, model_data in models.items():
                if isinstance(model_data, dict) and "model" in model_data:
                    validated_models[model_type] = model_data
                else:
                    self.logger.warning(f"⚠️ Invalid model structure for {model_type}")
            
            if not validated_models:
                self.logger.error("❌ No valid models found")
                return None
            
            self.logger.info(f"✅ Models loaded: {list(validated_models.keys())}")
            return validated_models
            
        except Exception as e:
            self.logger.exception(f"❌ Models loading failed: {e}")
            return None
    
    async def _define_parameter_spaces(self, models: Dict[str, Any]) -> Optional[Dict[str, Dict[str, List[Any]]]]:
        """Define parameter spaces for optimization."""
        try:
            self.logger.info("🔧 Defining parameter spaces...")
            
            parameter_spaces = {}
            
            # Define parameter spaces for different model types
            for model_type, model_data in models.items():
                model_class = model_data.get("model_class", "unknown")
                
                if "RandomForest" in model_class or "random_forest" in model_type:
                    parameter_spaces[model_type] = {
                        "n_estimators": [50, 100, 200, 300],
                        "max_depth": [None, 10, 20, 30],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4],
                        "max_features": ["sqrt", "log2", None]
                    }
                
                elif "XGBoost" in model_class or "xgboost" in model_type:
                    parameter_spaces[model_type] = {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [3, 6, 9],
                        "learning_rate": [0.01, 0.1, 0.2],
                        "subsample": [0.8, 0.9, 1.0],
                        "colsample_bytree": [0.8, 0.9, 1.0]
                    }
                
                elif "SVM" in model_class or "svm" in model_type:
                    parameter_spaces[model_type] = {
                        "C": [0.1, 1, 10, 100],
                        "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
                        "kernel": ["rbf", "poly", "sigmoid"]
                    }
                
                elif "LogisticRegression" in model_class or "logistic" in model_type:
                    parameter_spaces[model_type] = {
                        "C": [0.01, 0.1, 1, 10, 100],
                        "penalty": ["l1", "l2", "elasticnet"],
                        "solver": ["liblinear", "saga"]
                    }
                
                else:
                    # Default parameter space
                    parameter_spaces[model_type] = {
                        "random_state": [self.random_state],
                        "n_jobs": [-1]
                    }
            
            if not parameter_spaces:
                self.logger.error("❌ No parameter spaces defined")
                return None
            
            self.logger.info(f"✅ Parameter spaces defined for {len(parameter_spaces)} model types")
            return parameter_spaces
            
        except Exception as e:
            self.logger.exception(f"❌ Parameter space definition failed: {e}")
            return None
    
    @protect_optimisation_operation(ValidationLevel.CRITICAL)
    async def _perform_optimization(self, 
                                  training_data: Dict[str, Any],
                                  models: Dict[str, Any],
                                  parameter_spaces: Dict[str, Dict[str, List[Any]]]) -> Optional[Dict[str, Any]]:
        """Perform parameter optimization."""
        try:
            self.logger.info("🔧 Performing parameter optimization...")
            
            X = training_data["X"]
            y = training_data["y"]
            
            optimization_results = {
                "model_optimizations": {},
                "best_overall_model": None,
                "best_overall_score": -np.inf,
                "optimization_metadata": {
                    "n_samples": len(X),
                    "n_features": X.shape[1] if X.ndim > 1 else 1,
                    "optimization_timestamp": get_current_datetime().isoformat(),
                    "methods_tested": self.optimization_methods,
                    "cv_folds": self.cv_folds
                }
            }
            
            # Optimize each model type
            for model_type, model_data in models.items():
                try:
                    self.logger.info(f"🔧 Optimizing {model_type}...")
                    
                    if model_type not in parameter_spaces:
                        self.logger.warning(f"⚠️ No parameter space defined for {model_type}, skipping")
                        continue
                    
                    model_result = await self._optimize_model(
                        model_data, parameter_spaces[model_type], X, y
                    )
                    
                    if model_result is not None:
                        optimization_results["model_optimizations"][model_type] = model_result
                        
                        # Update best overall model
                        if model_result["best_score"] > optimization_results["best_overall_score"]:
                            optimization_results["best_overall_score"] = model_result["best_score"]
                            optimization_results["best_overall_model"] = model_type
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Optimization failed for {model_type}: {e}")
                    continue
            
            # Validate that at least one model was optimized
            if not optimization_results["model_optimizations"]:
                self.logger.error("❌ No models were successfully optimized")
                return None
            
            # Calculate overall optimization metrics
            optimization_results["overall_metrics"] = await self._calculate_overall_optimization_metrics(
                optimization_results
            )
            
            self.logger.info(f"✅ Parameter optimization completed: best model = {optimization_results['best_overall_model']}")
            return optimization_results
            
        except Exception as e:
            self.logger.exception(f"❌ Parameter optimization failed: {e}")
            return None
    
    async def _optimize_model(self, 
                            model_data: Dict[str, Any],
                            parameter_space: Dict[str, List[Any]],
                            X: np.ndarray,
                            y: np.ndarray) -> Optional[Dict[str, Any]]:
        """Optimize a single model."""
        try:
            model_class = model_data.get("model_class")
            base_model = model_data.get("model")
            
            if base_model is None:
                self.logger.error("❌ No base model found in model data")
                return None
            
            # Test different optimization methods
            method_results = {}
            
            for method in self.optimization_methods:
                try:
                    self.logger.info(f"🔧 Testing optimization method: {method}")
                    
                    method_result = await self._optimize_with_method(
                        base_model, parameter_space, X, y, method
                    )
                    
                    if method_result is not None:
                        method_results[method] = method_result
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Optimization method {method} failed: {e}")
                    continue
            
            if not method_results:
                self.logger.error("❌ All optimization methods failed")
                return None
            
            # Find best method
            best_method = max(method_results.keys(), key=lambda k: method_results[k]["best_score"])
            best_result = method_results[best_method]
            
            result = {
                "model_class": model_class,
                "best_method": best_method,
                "best_score": best_result["best_score"],
                "best_params": best_result["best_params"],
                "best_estimator": best_result["best_estimator"],
                "method_results": method_results,
                "optimization_time": time.time()
            }
            
            self.logger.info(f"✅ Model optimization completed: best method = {best_method}, score = {best_result['best_score']:.4f}")
            return result
            
        except Exception as e:
            self.logger.exception(f"❌ Model optimization failed: {e}")
            return None
    
    async def _optimize_with_method(self, 
                                  base_model: Any,
                                  parameter_space: Dict[str, List[Any]],
                                  X: np.ndarray,
                                  y: np.ndarray,
                                  method: str) -> Optional[Dict[str, Any]]:
        """Optimize using a specific method."""
        try:
            if method == "grid_search":
                return await self._grid_search_optimization(base_model, parameter_space, X, y)
            elif method == "random_search":
                return await self._random_search_optimization(base_model, parameter_space, X, y)
            elif method == "bayesian":
                return await self._bayesian_optimization(base_model, parameter_space, X, y)
            else:
                self.logger.error(f"❌ Unknown optimization method: {method}")
                return None
                
        except Exception as e:
            self.logger.exception(f"❌ Optimization method {method} failed: {e}")
            return None
    
    async def _grid_search_optimization(self, 
                                      base_model: Any,
                                      parameter_space: Dict[str, List[Any]],
                                      X: np.ndarray,
                                      y: np.ndarray) -> Optional[Dict[str, Any]]:
        """Perform grid search optimization."""
        try:
            from sklearn.model_selection import GridSearchCV
            
            # Limit parameter combinations for performance
            limited_space = {}
            for param, values in parameter_space.items():
                if len(values) > 3:
                    limited_space[param] = values[:3]  # Take first 3 values
                else:
                    limited_space[param] = values
            
            grid_search = GridSearchCV(
                base_model,
                limited_space,
                cv=self.cv_folds,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X, y)
            
            return {
                "best_score": grid_search.best_score_,
                "best_params": grid_search.best_params_,
                "best_estimator": grid_search.best_estimator_,
                "cv_results": grid_search.cv_results_
            }
            
        except Exception as e:
            self.logger.exception(f"❌ Grid search optimization failed: {e}")
            return None
    
    async def _random_search_optimization(self, 
                                        base_model: Any,
                                        parameter_space: Dict[str, List[Any]],
                                        X: np.ndarray,
                                        y: np.ndarray) -> Optional[Dict[str, Any]]:
        """Perform random search optimization."""
        try:
            from sklearn.model_selection import RandomizedSearchCV
            
            # Convert lists to distributions for random search
            param_distributions = {}
            for param, values in parameter_space.items():
                if all(isinstance(v, (int, float)) for v in values):
                    param_distributions[param] = values
                else:
                    param_distributions[param] = values
            
            random_search = RandomizedSearchCV(
                base_model,
                param_distributions,
                n_iter=min(self.n_trials, 50),  # Limit iterations
                cv=self.cv_folds,
                scoring='accuracy',
                n_jobs=-1,
                random_state=self.random_state,
                verbose=1
            )
            
            random_search.fit(X, y)
            
            return {
                "best_score": random_search.best_score_,
                "best_params": random_search.best_params_,
                "best_estimator": random_search.best_estimator_,
                "cv_results": random_search.cv_results_
            }
            
        except Exception as e:
            self.logger.exception(f"❌ Random search optimization failed: {e}")
            return None
    
    async def _bayesian_optimization(self, 
                                   base_model: Any,
                                   parameter_space: Dict[str, List[Any]],
                                   X: np.ndarray,
                                   y: np.ndarray) -> Optional[Dict[str, Any]]:
        """Perform Bayesian optimization."""
        try:
            # For now, fall back to random search if Bayesian optimization is not available
            self.logger.warning("⚠️ Bayesian optimization not implemented, falling back to random search")
            return await self._random_search_optimization(base_model, parameter_space, X, y)
            
        except Exception as e:
            self.logger.exception(f"❌ Bayesian optimization failed: {e}")
            return None
    
    async def _calculate_overall_optimization_metrics(self, 
                                                    optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall optimization metrics."""
        try:
            model_optimizations = optimization_results["model_optimizations"]
            
            # Calculate metrics across all models
            all_scores = [result["best_score"] for result in model_optimizations.values()]
            
            overall_metrics = {
                "n_models_optimized": len(model_optimizations),
                "best_score": max(all_scores),
                "worst_score": min(all_scores),
                "average_score": np.mean(all_scores),
                "score_std": np.std(all_scores),
                "improvement_over_baseline": max(all_scores) - 0.5,  # Assuming 0.5 baseline
                "optimization_success_rate": len(model_optimizations) / len(optimization_results.get("optimization_metadata", {}).get("methods_tested", 1))
            }
            
            return overall_metrics
            
        except Exception as e:
            self.logger.exception(f"❌ Overall optimization metrics calculation failed: {e}")
            return {}
    
    @protect_data_operation(ValidationLevel.STANDARD)
    async def _validate_optimization_results(self, optimization_results: Dict[str, Any]) -> bool:
        """Validate optimization results."""
        try:
            self.logger.info("🔍 Validating optimization results...")
            
            # Check required fields
            required_fields = ["model_optimizations", "best_overall_model", "best_overall_score", "overall_metrics"]
            missing_fields = [field for field in required_fields if field not in optimization_results]
            if missing_fields:
                self.logger.error(f"❌ Missing required fields: {missing_fields}")
                return False
            
            # Validate model optimizations
            if not optimization_results["model_optimizations"]:
                self.logger.error("❌ No model optimizations found")
                return False
            
            # Validate best model
            if optimization_results["best_overall_model"] not in optimization_results["model_optimizations"]:
                self.logger.error("❌ Best overall model not found in model optimizations")
                return False
            
            # Validate scores
            best_score = optimization_results["best_overall_score"]
            if best_score < 0.5:
                self.logger.warning("⚠️ Low optimization score detected")
            
            # Check for improvement
            overall_metrics = optimization_results["overall_metrics"]
            improvement = overall_metrics.get("improvement_over_baseline", 0)
            if improvement < self.min_improvement:
                self.logger.warning(f"⚠️ Minimal improvement detected: {improvement:.3f} < {self.min_improvement}")
            
            self.logger.info("✅ Optimization results validation passed")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Optimization results validation failed: {e}")
            return False
    
    @protect_data_operation(ValidationLevel.STANDARD, backup_enabled=True)
    async def _save_optimization_results(self, 
                                       optimization_results: Dict[str, Any],
                                       symbol: str,
                                       exchange: str,
                                       data_dir: str) -> bool:
        """Save optimization results."""
        try:
            self.logger.info("💾 Saving optimization results...")
            
            # Prepare save data
            save_data = {
                "optimization_results": optimization_results,
                "metadata": {
                    "symbol": symbol,
                    "exchange": exchange,
                    "timestamp": get_current_datetime().isoformat(),
                    "version": "enhanced_v1.0"
                }
            }
            
            # Save main results file
            results_file = f"{data_dir}/{exchange}_{symbol}_optimized_parameters.pkl"
            success = self.data_access.secure_data_saving(
                save_data,
                results_file,
                user_id="parameter_optimization",
                backup_existing=True
            )
            
            if not success:
                self.logger.error(f"❌ Failed to save optimization results to {results_file}")
                return False
            
            # Save metadata file
            metadata_file = f"{data_dir}/{exchange}_{symbol}_optimization_results.json"
            metadata = {
                "symbol": symbol,
                "exchange": exchange,
                "timestamp": get_current_datetime().isoformat(),
                "best_model": optimization_results["best_overall_model"],
                "best_score": optimization_results["best_overall_score"],
                "overall_metrics": optimization_results["overall_metrics"],
                "optimization_successful": True
            }
            
            success = self.data_access.secure_data_saving(
                metadata,
                metadata_file,
                user_id="parameter_optimization",
                backup_existing=True
            )
            
            if not success:
                self.logger.error(f"❌ Failed to save optimization metadata to {metadata_file}")
                return False
            
            # Save detailed results file
            results_json_file = f"{data_dir}/{exchange}_{symbol}_optimization_metrics.json"
            success = self.data_access.secure_data_saving(
                optimization_results,
                results_json_file,
                user_id="parameter_optimization",
                backup_existing=True
            )
            
            if not success:
                self.logger.error(f"❌ Failed to save detailed optimization results to {results_json_file}")
                return False
            
            self.logger.info("✅ Optimization results saved successfully")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Optimization results saving failed: {e}")
            return False