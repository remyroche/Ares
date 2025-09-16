"""
Enhanced HMM Models Training

Streamlined, robust, and well-reported HMM models training with comprehensive error handling.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
import json
import psutil
import os
import gc

warnings.filterwarnings('ignore')

# Core imports
from src.utils.tprint import tprint
from src.utils.ml_common.config.base_training_config import HMMTrainingConfig
from src.utils.ml_common.training.base_training_step import BaseTrainingStep

# Shared utilities
from .shared_utilities import (
    TrainingErrorHandler,
    UnifiedModelFactory,
    CircuitBreaker,
    ValidationUtils,
    ProgressReporter,
    MemoryTracker
)
# Using tprint for all logging - no logger needed


@dataclass
class TrainingMetrics:
    """Enhanced training metrics container with additional monitoring."""
    accuracy: float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    training_time: float = 0.0
    convergence_epochs: int = 0
    memory_usage_mb: float = 0.0
    validation_loss: Optional[float] = None
    test_accuracy: Optional[float] = None
    error_message: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class ModelResult:
    """Enhanced model result container with additional metadata."""
    model: Any
    metrics: TrainingMetrics
    feature_importance: Optional[Dict[str, float]] = None
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    training_history: Optional[Dict[str, List[float]]] = None


class CircuitBreaker:
    """Circuit breaker to prevent cascading failures in model training."""
    
    def __init__(self, failure_threshold: int = 3, timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
                tprint("🔄 Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN - too many failures detected")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                tprint("✅ Circuit breaker reset to CLOSED")
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                tprint(f"🚨 Circuit breaker opened after {self.failure_count} failures")
            
            raise e


class TrainingErrorHandler:
    """Centralized error handling for training operations."""
    
    @staticmethod
    def handle_model_creation_error(model_type: str, error: Exception) -> ModelResult:
        """Standardized model creation error handling."""
        return ModelResult(
            model=None,
            metrics=TrainingMetrics(
                error_message=f"Failed to create {model_type}: {str(error)}",
                training_time=0.0
            )
        )
    
    @staticmethod
    def handle_training_error(model_type: str, error: Exception, training_time: float) -> ModelResult:
        """Standardized training error handling."""
        return ModelResult(
            model=None,
            metrics=TrainingMetrics(
                error_message=f"Failed to train {model_type}: {str(error)}",
                training_time=training_time
            )
        )


logger = system_logger.getChild('HMMModelsTrainingEnhanced')


# Import shared data classes
from .shared_utilities.training_error_handler import TrainingMetrics, ModelResult

class HMMModelsTrainingEnhanced(BaseTrainingStep):
    """
    Enhanced HMM Models Training with streamlined code, robust error handling, and comprehensive reporting.
    """
    
    def __init__(self, config: Optional[Union[HMMTrainingConfig, Dict[str, Any]]] = None):
        """
        Initialize enhanced HMM models training.

        Args:
            config: HMM training configuration object or dictionary of parameters
        """
        if config is None:
            config = HMMTrainingConfig(
                model_name="hmm_models_enhanced",
                timeframe="1h",
                n_features=100,
                sequence_length=20,
                n_regimes=3,
                model_types=["lightgbm", "elastic_net_lr", "xgboost"],
                hpo_trials=50,
                enable_multi_objective=True
            )
        elif isinstance(config, dict):
            # Convert dictionary to HMMTrainingConfig
            default_config = HMMTrainingConfig()
            config_dict = {**default_config.__dict__, **config}
            config = HMMTrainingConfig(**config_dict)

        # Validate configuration before proceeding
        self._validate_config(config)

        super().__init__(config)
        self.logger = logger.getChild('HMMModelsTrainingEnhanced')

        self.circuit_breaker = CircuitBreaker(failure_threshold=2, timeout=60)
        self.progress_reporter = None
        self.memory_tracker = MemoryTracker()
        
        # Initialize components with error handling
        self._initialize_components()
        
        # Training state
        self.training_start_time = None
        self.training_results = {}
        
        tprint("✅ Enhanced HMM Models Training initialized with circuit breaker protection")
    
    def _validate_config(self, config: HMMTrainingConfig) -> None:
        """
        Validate configuration parameters with fast-fail on critical errors.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        try:
            # Use shared validation utility
            if not ValidationUtils.validate_config(config):
                raise ValueError("Configuration validation failed")
            
            # Additional HMM-specific validations
            warnings = []
            
            # Warning validations (don't cause fast-fail)
            if config.n_features > 1000:
                warnings.append("WARNING: Large number of features may impact performance")
            
            if config.hpo_trials > 1000:
                warnings.append("WARNING: Large number of HPO trials may take very long")
            
            if config.sequence_length > 100:
                warnings.append("WARNING: Large sequence length may impact memory usage")
            
            # Log warnings
            if warnings:
                for warning in warnings:
                    self.logger.warning(f"⚠️ {warning}")
            
            self.logger.info("✅ Configuration validation passed")
            
        except Exception as e:
            self.logger.error(f"❌ Configuration validation error: {e}")
            raise ValueError(f"Configuration validation failed: {e}") from e
    
    def _initialize_components(self) -> None:
        """Initialize training components with comprehensive error handling."""
        # Initialize feature generator with specific error handling
        self.feature_generator = self._initialize_feature_generator()
        
        # Initialize feature selector with specific error handling
        self.feature_selector = self._initialize_feature_selector()
        
        # Initialize evaluation utilities with specific error handling
        self.evaluation_utils = self._initialize_evaluation_utils()

        # Model creation now handled by shared UnifiedModelFactory
    
    def _initialize_feature_generator(self) -> Optional[Any]:
        """Initialize feature generator with specific error handling."""
        try:
            # Try primary import
            from src.feature_engineering.feature_generators import FeatureGenerators
            generator = FeatureGenerators()
            self.logger.info("✅ Feature generator initialized from feature_engineering")
            return generator
        except ImportError as primary_error:
            self.logger.debug(f"Primary feature generator import failed: {primary_error}")
            try:
                # Fallback to standalone compatibility
                from src.hmm_feature_compatibility import FeatureGenerators
                generator = FeatureGenerators()
                self.logger.info("✅ Feature generator initialized from standalone compatibility")
                return generator
            except ImportError as fallback_error:
                self.logger.warning(f"⚠️ Feature generator not available - primary: {primary_error}, fallback: {fallback_error}")
                return None
        except Exception as e:
            self.logger.error(f"❌ Unexpected error initializing feature generator: {e}")
            return None
    
    def _initialize_feature_selector(self) -> Optional[Any]:
        """Initialize feature selector with specific error handling."""
        try:
            from src.training.utils.feature_selection.main_framework import FeatureSelectionFramework
            fs_config = {
                'selection_methods': ['mrmr', 'lasso_stability'],
                'max_features': self.config.n_features,
                'enable_stability_analysis': True
            }
            selector = FeatureSelectionFramework(fs_config)
            self.logger.info("✅ Feature selector initialized")
            return selector
        except ImportError as e:
            self.logger.warning(f"⚠️ Feature selector not available: {e}")
            return None
        except Exception as e:
            self.logger.error(f"❌ Unexpected error initializing feature selector: {e}")
            return None
    
    def _initialize_evaluation_utils(self) -> Optional[Any]:
        """Initialize evaluation utilities with specific error handling."""
        try:
            from src.utils.ml_common.evaluation.evaluation_utils import EvaluationUtils
            utils = EvaluationUtils()
            self.logger.info("✅ Evaluation utilities initialized")
            return utils
        except ImportError as e:
            self.logger.warning(f"⚠️ Evaluation utilities not available: {e}")
            return None
        except Exception as e:
            self.logger.error(f"❌ Unexpected error initializing evaluation utilities: {e}")
            return None
    
    def _convert_to_numpy_array(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Convert data to numpy array with proper validation and error handling.
        
        Args:
            data: Input data (DataFrame or numpy array)
            
        Returns:
            numpy array
            
        Raises:
            ValueError: If conversion fails or data is invalid
        """
        try:
            if isinstance(data, np.ndarray):
                # Already a numpy array, validate it
                if data.size == 0:
                    raise ValueError("Input array is empty")
                if np.any(np.isnan(data)):
                    raise ValueError("Input array contains NaN values")
                if np.any(np.isinf(data)):
                    raise ValueError("Input array contains infinite values")
                return data
            
            elif isinstance(data, pd.DataFrame):
                # Convert DataFrame to numpy array
                if data.empty:
                    raise ValueError("Input DataFrame is empty")
                
                # Check for non-numeric columns
                numeric_data = data.select_dtypes(include=[np.number])
                if numeric_data.empty:
                    raise ValueError("DataFrame contains no numeric columns")
                
                if len(numeric_data.columns) != len(data.columns):
                    non_numeric_cols = set(data.columns) - set(numeric_data.columns)
                    self.logger.warning(f"⚠️ Dropping non-numeric columns: {non_numeric_cols}")
                
                # Convert to numpy array
                array_data = numeric_data.values
                
                # Validate the converted array
                if np.any(np.isnan(array_data)):
                    nan_count = np.isnan(array_data).sum()
                    self.logger.warning(f"⚠️ Converted array contains {nan_count} NaN values")
                
                if np.any(np.isinf(array_data)):
                    inf_count = np.isinf(array_data).sum()
                    self.logger.warning(f"⚠️ Converted array contains {inf_count} infinite values")
                
                return array_data
            
            else:
                raise ValueError(f"Unsupported data type: {type(data)}. Expected numpy array or DataFrame.")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to convert data to numpy array: {e}")
            raise ValueError(f"Data conversion failed: {e}") from e
    
# Model registration now handled by shared UnifiedModelFactory
    
    def _validate_input_data(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray) -> bool:
        """
        Enhanced input validation with early exit on critical failures.
        
        Args:
            X: Input features
            y: Target values
            regime_labels: Regime labels
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            # Use shared validation utilities
            if not ValidationUtils.validate_data_shapes(X, y, regime_labels):
                return False
            
            if not ValidationUtils.validate_data_quality(X, y, regime_labels):
                return False
            
            if not ValidationUtils.validate_regime_distribution(regime_labels, min_samples_per_regime=10):
                return False
            
            # Additional HMM-specific validations
            warnings = []
            unique_regimes = np.unique(regime_labels)
            if len(unique_regimes) < 2:
                critical_failures.append(f"Need at least 2 regimes, found {len(unique_regimes)}")
            
            # Early exit on critical failures
            if critical_failures:
                tprint(f"❌ Critical validation failures: {critical_failures}")
                return False
            
            # Warning checks (don't cause early exit)
            if len(X) < 1000:
                warnings.append(f"Small dataset: {len(X)} samples (recommended: >1000)")
            
            # Check minimum samples per regime
            for regime in unique_regimes:
                regime_count = np.sum(regime_labels == regime)
                if regime_count < 10:  # Minimum samples per regime
                    warnings.append(f"Regime {regime} has only {regime_count} samples (minimum: 10)")
            
            # Log warnings
            if warnings:
                for warning in warnings:
                    tprint(f"⚠️ {warning}")
            
            tprint(f"✅ Enhanced validation passed: {len(X)} samples, {len(unique_regimes)} regimes")
            return True
            
        except Exception as e:
            tprint(f"❌ Validation error: {e}")
            return False
    
    def _prepare_features(self, X: Union[np.ndarray, pd.DataFrame], feature_names: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare and enhance features with comprehensive error handling.
        
        Args:
            X: Input features
            feature_names: Optional feature names
            
        Returns:
            Tuple of (enhanced_features, feature_names)
        """
        try:
            # Convert to DataFrame if needed
            if isinstance(X, np.ndarray):
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                X_df = pd.DataFrame(X, columns=feature_names)
            else:
                X_df = X.copy()
                if feature_names is None:
                    feature_names = list(X_df.columns)
            
            # Ensure only numeric columns
            numeric_columns = X_df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                raise ValueError("No numeric columns found in input data")
            
            X_numeric = X_df[numeric_columns]
            tprint(f"📊 Using {len(numeric_columns)} numeric features")
            
            # Enhance features if generator is available
            if self.feature_generator is not None:
                try:
                    X_enhanced = self.feature_generator.generate_features_for_hmm(X_numeric)
                    self.logger.info(f"✅ Enhanced features: {X_enhanced.shape[1]} total features")
                    return X_enhanced, list(X_enhanced.columns)
                except Exception as e:
                    self.logger.warning(f"⚠️ Feature enhancement failed: {e}, using original features")
                    return X_numeric, list(X_numeric.columns)
            else:
                return X_numeric, list(X_numeric.columns)
                
        except Exception as e:
            self.logger.error(f"❌ Feature preparation failed: {e}")
            raise
    
    def _select_features(self, X: pd.DataFrame, y: np.ndarray, is_classification: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select optimal features with comprehensive error handling.
        
        Args:
            X: Input features
            y: Target values
            is_classification: Whether this is classification
            
        Returns:
            Tuple of (selected_features, selected_feature_names)
        """
        try:
            if self.feature_selector is None:
                self.logger.warning("⚠️ Feature selector not available, using all features")
                return X, list(X.columns)
            
            # Validate shapes before proceeding - don't modify input data
            if len(X) != len(y):
                error_msg = f"Shape mismatch: X has {len(X)} samples, y has {len(y)} samples"
                self.logger.error(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            # Create copies to avoid modifying original data
            X_copy = X.copy()
            y_copy = y.copy()
            
            # Apply feature selection
            selection_result = self.feature_selector.select_features(
                X_copy, y_copy,
                method='comprehensive',
                max_features=self.config.n_features,
                is_classification=is_classification
            )
            
            selected_features = selection_result.get('selected_features', list(X.columns)[:self.config.n_features])
            
            # Validate selected features exist in the DataFrame
            missing_features = [f for f in selected_features if f not in X.columns]
            if missing_features:
                self.logger.warning(f"⚠️ Some selected features not found in DataFrame: {missing_features}")
                # Filter out missing features
                selected_features = [f for f in selected_features if f in X.columns]
            
            if not selected_features:
                self.logger.warning("⚠️ No valid features selected, using first n_features")
                selected_features = list(X.columns)[:self.config.n_features]
            
            X_selected = X[selected_features]
            
            self.logger.info(f"✅ Feature selection completed: {len(selected_features)} features selected")
            return X_selected, selected_features
            
        except Exception as e:
            self.logger.error(f"❌ Feature selection failed: {e}")
            # Fallback to basic selection with validation
            fallback_features = list(X.columns)[:self.config.n_features]
            if not fallback_features:
                raise ValueError("No features available for fallback selection")
            return X[fallback_features], fallback_features
    
    def _create_model(self, model_type: str, **kwargs) -> Any:
        """
        Create model instance using shared UnifiedModelFactory.
        
        Args:
            model_type: Type of model to create
            **kwargs: Additional model parameters
            
        Returns:
            Model instance
        """
        try:
            # Use shared unified model factory
            return UnifiedModelFactory.create_model(model_type, **kwargs)
        except Exception as e:
            self.logger.error(f"❌ Failed to create model {model_type}: {e}")
            raise
    
    def _train_single_model(self, model_type: str, X: np.ndarray, y: np.ndarray) -> ModelResult:
        """
        Train a single model with circuit breaker protection and enhanced error handling.
        
        Args:
            model_type: Type of model to train
            X: Training features
            y: Training targets
            
        Returns:
            ModelResult with trained model and metrics
        """
        start_time = time.time()
        metrics = TrainingMetrics()
        # Use shared memory tracker
        memory_tracker = MemoryTracker()
        
        try:
            self.logger.info(f"🔄 Training {model_type}...")
            memory_tracker.take_snapshot(f"{model_type}_start")
            
            # Create model with circuit breaker protection
            def create_and_train_model():
                model = self._create_model(model_type)
                memory_tracker.take_snapshot(f"{model_type}_model_created")
                
                # Train model
                model.fit(X, y)
                memory_tracker.take_snapshot(f"{model_type}_model_fitted")
                
                # Evaluate model
                predictions = model.predict(X)
                accuracy = np.mean(predictions == y)
                
                # Use evaluation utilities if available (preserving original functionality)
                if self.evaluation_utils is not None:
                    try:
                        eval_metrics = self.evaluation_utils.evaluate_model_performance(
                            model, X, y,
                            metrics=['accuracy', 'f1_score', 'precision', 'recall'],
                            is_classification=True
                        )
                        metrics.accuracy = eval_metrics.get('accuracy', accuracy)
                        metrics.f1_score = eval_metrics.get('f1_score', 0.0)
                        metrics.precision = eval_metrics.get('precision', 0.0)
                        metrics.recall = eval_metrics.get('recall', 0.0)
                    except Exception as e:
                        metrics.warnings.append(f"Evaluation utilities failed: {e}")
                        # Fallback to basic metrics
                        metrics.accuracy = accuracy
                        try:
                            from sklearn.metrics import f1_score, precision_score, recall_score
                            metrics.f1_score = f1_score(y, predictions, average='weighted')
                            metrics.precision = precision_score(y, predictions, average='weighted')
                            metrics.recall = recall_score(y, predictions, average='weighted')
                        except Exception as e2:
                            metrics.warnings.append(f"Fallback metrics calculation failed: {e2}")
                else:
                    # Fallback evaluation (preserving original functionality)
                    try:
                        from sklearn.metrics import f1_score, precision_score, recall_score
                        metrics.accuracy = accuracy
                        metrics.f1_score = f1_score(y, predictions, average='weighted')
                        metrics.precision = precision_score(y, predictions, average='weighted')
                        metrics.recall = recall_score(y, predictions, average='weighted')
                    except Exception as e:
                        metrics.warnings.append(f"Fallback metrics calculation failed: {e}")
                        metrics.accuracy = accuracy
                
                # Get feature importance
                feature_importance = None
                try:
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = dict(zip(range(len(model.feature_importances_)), model.feature_importances_))
                    elif hasattr(model, 'coef_'):
                        feature_importance = dict(zip(range(len(model.coef_[0])), np.abs(model.coef_[0])))
                except Exception as e:
                    self.logger.debug(f"Feature importance not available for {model_type}: {e}")
                
                # Get hyperparameters
                hyperparameters = None
                try:
                    if hasattr(model, 'get_params'):
                        hyperparameters = model.get_params()
                except Exception as e:
                    self.logger.debug(f"Could not get hyperparameters: {e}")
                
                return model, predictions, feature_importance, hyperparameters
            
            # Execute with circuit breaker protection
            model, predictions, feature_importance, hyperparameters = self.circuit_breaker.call(create_and_train_model)
            
            training_time = time.time() - start_time
            metrics.training_time = training_time
            
            # Calculate memory usage
            memory_tracker.take_snapshot(f"{model_type}_completed")
            metrics.memory_usage_mb = memory_tracker.get_memory_increase()
            
            # Get probabilities if available
            probabilities = None
            try:
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X)
            except Exception as e:
                metrics.warnings.append(f"Could not get probabilities: {e}")
            
            self.logger.info(f"✅ {model_type} trained successfully (accuracy: {metrics.accuracy:.4f}, time: {training_time:.2f}s, memory: {metrics.memory_usage_mb:.1f}MB)")
            
            # Cleanup memory
            memory_tracker.cleanup()
            
            return ModelResult(
                model=model,
                metrics=metrics,
                feature_importance=feature_importance,
                predictions=predictions,
                probabilities=probabilities,
                hyperparameters=hyperparameters
            )
            
        except Exception as e:
            training_time = time.time() - start_time
            metrics.training_time = training_time
            metrics.error_message = str(e)
            
            # Calculate memory usage even on failure
            memory_tracker.take_snapshot(f"{model_type}_failed")
            metrics.memory_usage_mb = memory_tracker.get_memory_increase()
            
            self.logger.error(f"❌ Failed to train {model_type}: {e}")
            
            # Cleanup memory
            memory_tracker.cleanup()
            
            # Use centralized error handler
            return TrainingErrorHandler.handle_training_error(model_type, e, training_time)
    
    def _generate_comprehensive_report(self, results: Dict[str, Any], execution_time: float) -> Dict[str, Any]:
        """
        Generate comprehensive training report with real metrics.
        
        Args:
            results: Training results
            execution_time: Total execution time
            
        Returns:
            Comprehensive report dictionary
        """
        try:
            report = {
                "report_type": "HMM Models Training Enhanced Report",
                "timestamp": pd.Timestamp.now().isoformat(),
                "execution_summary": {
                    "total_execution_time": execution_time,
                    "models_trained": len(results.get('model_results', {})),
                    "successful_models": sum(1 for r in results.get('model_results', {}).values() 
                                           if r.metrics.error_message is None),
                    "failed_models": sum(1 for r in results.get('model_results', {}).values() 
                                        if r.metrics.error_message is not None),
                    "circuit_breaker_state": self.circuit_breaker.state,
                    "circuit_breaker_failures": self.circuit_breaker.failure_count
                },
                "model_performance": {},
                "feature_analysis": {
                    "total_features": results.get('total_features', 0),
                    "selected_features": results.get('selected_features', 0),
                    "feature_selection_ratio": results.get('selected_features', 0) / max(results.get('total_features', 1), 1)
                },
                "regime_analysis": {
                    "total_regimes": results.get('n_regimes', 0),
                    "regime_distribution": results.get('regime_distribution', {})
                },
                "computational_metrics": {
                    "average_training_time": np.mean([r.metrics.training_time for r in results.get('model_results', {}).values()]),
                    "total_memory_usage": sum([r.metrics.memory_usage_mb for r in results.get('model_results', {}).values()]),
                    "training_efficiency": results.get('selected_features', 0) / max(execution_time, 0.001)
                },
                "recommendations": []
            }
            
            # Analyze model performance and collect warnings
            model_results = results.get('model_results', {})
            all_warnings = []
            
            if model_results:
                best_accuracy = -1
                best_model = None
                accuracies = []
                
                for model_name, model_result in model_results.items():
                    metrics = model_result.metrics
                    
                    # Collect warnings
                    if metrics.warnings:
                        all_warnings.extend([f"{model_name}: {w}" for w in metrics.warnings])
                    
                    report["model_performance"][model_name] = {
                        "accuracy": metrics.accuracy,
                        "f1_score": metrics.f1_score,
                        "precision": metrics.precision,
                        "recall": metrics.recall,
                        "training_time": metrics.training_time,
                        "status": "success" if metrics.error_message is None else "failed",
                        "error": metrics.error_message,
                        "warnings": metrics.warnings
                    }
                    
                    if metrics.error_message is None:
                        accuracies.append(metrics.accuracy)
                        if metrics.accuracy > best_accuracy:
                            best_accuracy = metrics.accuracy
                            best_model = model_name
                
                # Add performance summary
                if accuracies:
                    report["performance_summary"] = {
                        "best_model": best_model,
                        "best_accuracy": best_accuracy,
                        "average_accuracy": np.mean(accuracies),
                        "accuracy_std": np.std(accuracies),
                        "performance_variance": np.var(accuracies)
                    }
            
            # Add warnings to report
            report["warnings"] = list(set(all_warnings))  # Remove duplicates
            
            # Generate enhanced recommendations
            recommendations = []
            
            if report["execution_summary"]["failed_models"] > 0:
                recommendations.append(f"Address {report['execution_summary']['failed_models']} failed model(s)")
            
            if report["execution_summary"]["circuit_breaker_state"] == "OPEN":
                recommendations.append("Circuit breaker is OPEN - investigate systematic failures")
            
            if len(all_warnings) > 0:
                recommendations.append(f"Address {len(all_warnings)} warnings for better performance")
            
            if report["performance_summary"]["average_accuracy"] < 0.7:
                recommendations.append("Consider feature engineering or data preprocessing improvements")
            
            if report["computational_metrics"]["average_training_time"] > 60:
                recommendations.append("Consider reducing model complexity or using faster algorithms")
            
            if report["feature_analysis"]["feature_selection_ratio"] > 0.5:
                recommendations.append("High feature selection ratio - consider more aggressive feature selection")
            
            report["recommendations"] = recommendations
            
            self.logger.info("✅ Comprehensive report generated successfully")
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate comprehensive report: {e}")
            return {
                "report_type": "HMM Models Training Report (Error)",
                "error": str(e),
                "timestamp": pd.Timestamp.now().isoformat(),
                "status": "Report generation failed"
            }
    
    def execute(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        hmm_states: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute enhanced HMM models training with comprehensive error handling and reporting.
        
        Args:
            X: Input features
            y: Target values
            regime_labels: Regime labels for each sample
            feature_names: Names of input features
            hmm_states: HMM cluster/regime states
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing training results and comprehensive report
        """
        self.logger.info("🚀 Starting Enhanced HMM Models Training")
        self.training_start_time = time.time()
        
        try:
            # Step 1: Input validation
            self.logger.info("🔄 Step 1: Validating inputs...")
            if not self._validate_input_data(X, y, regime_labels):
                raise ValueError("Input validation failed")
            
            # Step 2: Feature preparation
            self.logger.info("🔄 Step 2: Preparing features...")
            X_enhanced, enhanced_feature_names = self._prepare_features(X, feature_names)
            
            # Step 3: Feature selection
            self.logger.info("🔄 Step 3: Selecting features...")
            X_selected, selected_features = self._select_features(
                X_enhanced, y, 
                is_classification=kwargs.get('is_classification', True)
            )
            
            # Step 4: Initialize progress reporter and train models
            self.logger.info("🔄 Step 4: Training models with real-time progress...")
            self.progress_reporter = ProgressReporter(len(self.config.model_types))
            model_results = {}
            
            for model_type in self.config.model_types:
                try:
                    # Convert to numpy array for training with proper validation
                    X_train = self._convert_to_numpy_array(X_selected)
                    
                    # Train model
                    model_result = self._train_single_model(model_type, X_train, y)
                    model_results[model_type] = model_result
                    
                    # Update progress
                    success = model_result.metrics.error_message is None
                    accuracy = model_result.metrics.accuracy if success else None
                    error_message = model_result.metrics.error_message if not success else None
                    self.progress_reporter.update_progress(
                        model_type, success, model_result.metrics.training_time, 
                        accuracy, error_message
                    )
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to train {model_type}: {e}")
                    # Create failed result using centralized error handler
                    model_results[model_type] = TrainingErrorHandler.handle_training_error(model_type, e, 0.0)
                    self.progress_reporter.update_progress(model_type, False, 0.0, None, str(e))
            
            # Step 5: Finish progress reporting
            self.progress_reporter.finish_report()
            
            # Step 6: Analyze regimes
            unique_regimes, regime_counts = np.unique(regime_labels, return_counts=True)
            regime_distribution = {
                f"regime_{regime}": {
                    "count": int(count),
                    "percentage": float(count / len(regime_labels) * 100)
                }
                for regime, count in zip(unique_regimes, regime_counts)
            }
            
            # Step 7: Create final results with proper artifact formatting
            execution_time = time.time() - self.training_start_time
            
            # Format model results for artifacts
            hmm_base_models = []
            hmm_training_metrics = {}
            hmm_model_performance = {}
            
            for model_name, model_result in model_results.items():
                if model_result.model is not None:
                    # Add model to base models list
                    hmm_base_models.append({
                        'model_name': model_name,
                        'model_type': model_name,
                        'model_object': model_result.model,
                        'hyperparameters': model_result.hyperparameters
                    })
                    
                    # Add training metrics
                    hmm_training_metrics[model_name] = {
                        'accuracy': model_result.metrics.accuracy,
                        'f1_score': model_result.metrics.f1_score,
                        'precision': model_result.metrics.precision,
                        'recall': model_result.metrics.recall,
                        'training_time': model_result.metrics.training_time,
                        'convergence_epochs': model_result.metrics.convergence_epochs,
                        'memory_usage_mb': model_result.metrics.memory_usage_mb,
                        'validation_loss': model_result.metrics.validation_loss,
                        'test_accuracy': model_result.metrics.test_accuracy,
                        'warnings': model_result.metrics.warnings
                    }
                    
                    # Add performance metrics
                    hmm_model_performance[model_name] = {
                        'feature_importance': model_result.feature_importance,
                        'predictions_available': model_result.predictions is not None,
                        'probabilities_available': model_result.probabilities is not None,
                        'training_history_available': model_result.training_history is not None
                    }
            
            results = {
                'model_results': model_results,
                'artifacts': {
                    'hmm_base_models': hmm_base_models,
                    'hmm_training_metrics': hmm_training_metrics,
                    'hmm_model_performance': hmm_model_performance
                },
                'metadata': {
                    'total_features': X_enhanced.shape[1],
                    'selected_features': len(selected_features),
                    'selected_feature_names': selected_features,
                    'n_regimes': len(unique_regimes),
                    'regime_distribution': regime_distribution,
                    'execution_time': execution_time,
                    'config': self.config,
                    'circuit_breaker_state': self.circuit_breaker.state,
                    'circuit_breaker_failures': self.circuit_breaker.failure_count,
                    'models_trained': len(hmm_base_models),
                    'successful_models': len([m for m in hmm_base_models if m['model_object'] is not None])
                },
                'training_time': execution_time
            }
            
            # Step 8: Generate comprehensive report
            self.logger.info("🔄 Step 8: Generating comprehensive report...")
            comprehensive_report = self._generate_comprehensive_report(results, execution_time)
            results['comprehensive_report'] = comprehensive_report
            
            # Step 9: Save results if configured
            if self.config.save_models:
                self.logger.info("🔄 Step 9: Saving models...")
                try:
                    symbol = kwargs.get('symbol', 'UNKNOWN')
                    exchange = kwargs.get('exchange', 'UNKNOWN')
                    timeframe = kwargs.get('timeframe', self.config.timeframe)
                    
                    # Save successful models only
                    successful_models = {
                        name: result.model for name, result in model_results.items()
                        if result.model is not None
                    }
                    
                    if successful_models:
                        saved_paths = self.save_models(
                            models=successful_models,
                            model_type=self.config.model_name,
                            symbol=symbol,
                            exchange=exchange,
                            timeframe=timeframe
                        )
                        results['saved_model_paths'] = saved_paths
                        self.logger.info(f"✅ Models saved: {len(saved_paths)} files")
                    else:
                        self.logger.warning("⚠️ No successful models to save")
                        
                except Exception as e:
                    self.logger.error(f"❌ Failed to save models: {e}")
                    results['save_error'] = str(e)
            
            # Log enhanced final summary
            successful_count = sum(1 for r in model_results.values() if r.metrics.error_message is None)
            self.logger.info(f"✅ Enhanced HMM Models Training completed: {successful_count}/{len(model_results)} models successful")
            self.logger.info(f"📊 Total execution time: {execution_time:.2f}s")
            self.logger.info(f"🔧 Circuit breaker state: {self.circuit_breaker.state}")
            if self.circuit_breaker.failure_count > 0:
                self.logger.info(f"⚠️ Circuit breaker failures: {self.circuit_breaker.failure_count}")
            
            return results
            
        except Exception as e:
            execution_time = time.time() - self.training_start_time if self.training_start_time else 0
            self.logger.error(f"❌ Enhanced HMM Models Training failed: {e}")
            
            return {
                'model_results': {},
                'metadata': {
                    'error': str(e),
                    'execution_time': execution_time,
                    'config': self.config
                },
                'training_time': execution_time,
                'comprehensive_report': {
                    "report_type": "HMM Models Training Report (Error)",
                    "error": str(e),
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "status": "Training failed"
                }
            }


# Convenience functions
def create_enhanced_hmm_models_training(
    config: Optional[HMMTrainingConfig] = None
) -> HMMModelsTrainingEnhanced:
    """Create enhanced HMM models training step."""
    return HMMModelsTrainingEnhanced(config)


def execute_enhanced_hmm_models_training(
    X: np.ndarray,
    y: np.ndarray,
    regime_labels: np.ndarray,
    config: Optional[HMMTrainingConfig] = None,
    feature_names: Optional[List[str]] = None,
    hmm_states: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Execute enhanced HMM models training step."""
    step = create_enhanced_hmm_models_training(config)
    return step.execute(X, y, regime_labels, feature_names, hmm_states)


# Example usage
if __name__ == "__main__":
    print("Enhanced HMM Models Training")
    print("=" * 50)
    
    # Create configuration
    config = HMMTrainingConfig(
        model_name="hmm_models_enhanced",
        timeframe="1h",
        n_features=50,
        sequence_length=20,
        n_regimes=3,
        model_types=["lightgbm", "elastic_net", "xgboost"],
        hpo_trials=25,
        enable_multi_objective=True
    )
    
    # Create training step
    training_step = create_enhanced_hmm_models_training(config)
    
    print(f"✅ Created enhanced training step with {len(config.model_types)} model types")
    print(f"📊 Features: {config.n_features}")
    print(f"📊 Sequence length: {config.sequence_length}")
    print(f"📊 HPO trials: {config.hpo_trials}")
    
    print("\n🎯 Key enhancements:")
    print("- ✅ Circuit breaker pattern prevents cascading failures")
    print("- ✅ Model factory pattern reduces code duplication")
    print("- ✅ Real-time progress reporting with ETA")
    print("- ✅ Enhanced input validation with early exit")
    print("- ✅ Centralized error handling")
    print("- ✅ Warning collection and reporting")
    print("- ✅ Comprehensive reporting with actionable insights")
    print("- ✅ Silent failure prevention")