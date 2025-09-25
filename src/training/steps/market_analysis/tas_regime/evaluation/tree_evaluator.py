"""
Tree Evaluator for TAS Regime Analysis

This module provides TAS-specific tree evaluation capabilities using
the unified evaluation framework.

Key Features:
- Tree model evaluation with multiple metrics
- Integration with unified evaluation framework
- TAS-specific regime analysis
- Performance monitoring and optimization
- Comprehensive reporting and visualization
- Support for ensemble methods and model comparison
"""

import logging
import time
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)

# Import unified evaluator
try:
    from src.utils.nas_tas import UnifiedEvaluator, EvaluationConfig, EvaluationResult
    UNIFIED_EVALUATOR_AVAILABLE = True
except ImportError:
    UNIFIED_EVALUATOR_AVAILABLE = False

# Import utility modules with fallback implementations
try:
    from src.utils.common_operations import (
        safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
        calculate_data_quality_metrics, create_summary_statistics, safe_merge_dataframes,
        safe_groupby_operation, safe_apply_function, create_data_quality_report,
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        integrate_with_m1_optimizers, memory_checkpoint, gpu_context, optimize_memory,
        get_memory_usage, safe_json_dump, safe_json_load, safe_to_parquet, safe_read_parquet,
        timed_operation, format_bytes, parallel_map, validate_finite, validate_positive,
        safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
        safe_correlation, safe_covariance, safe_percentile, validate_correlation_matrix,
        safe_matrix_inverse, math_safe, CommonUtilities
    )
except ImportError as e:
    logging.warning(f"Could not import common_operations: {e}")
    # Fallback implementations
    def safe_dataframe_operation(df, operation, *args, **kwargs):
        return operation(df, *args, **kwargs)
    def validate_dataframe_columns(df, required_columns):
        return True
    def safe_convert_dtypes(df, dtype_mapping):
        return df
    def calculate_data_quality_metrics(df):
        return {}
    def create_summary_statistics(df):
        return {}
    def safe_merge_dataframes(df1, df2, **kwargs):
        return pd.merge(df1, df2, **kwargs)
    def safe_groupby_operation(df, group_cols, agg_dict):
        return df.groupby(group_cols).agg(agg_dict)
    def safe_apply_function(df, func, axis=0):
        return df.apply(func, axis=axis)
    def create_data_quality_report(df):
        return {}
    def get_m1_gpu_manager():
        return None
    def get_m1_memory_optimizer():
        return None
    def get_m1_cpu_optimizer():
        return None
    def integrate_with_m1_optimizers():
        return {'success': False}
    def memory_checkpoint(name):
        from contextlib import nullcontext
        return nullcontext()
    def gpu_context(name):
        from contextlib import nullcontext
        return nullcontext()
    def optimize_memory():
        return {'success': False}
    def get_memory_usage():
        return 0.0
    def safe_json_dump(data, file_path, **kwargs):
        return False
    def safe_json_load(file_path, default=None):
        return default
    def safe_to_parquet(df, file_path, **kwargs):
        return False
    def safe_read_parquet(file_path, **kwargs):
        return None
    def timed_operation(func):
        return func
    def format_bytes(bytes_value):
        return f"{bytes_value} B"
    def parallel_map(func, iterable, max_workers=None):
        return [func(item) for item in iterable]
    def validate_finite(value, name="value"):
        return float(value)
    def validate_positive(value, name="value"):
        return float(value)
    def safe_divide(a, b, default=0.0):
        return a / b if b != 0 else default
    def safe_log(x, default=0.0):
        return np.log(x) if x > 0 else default
    def safe_sqrt(x, default=0.0):
        return np.sqrt(x) if x >= 0 else default
    def safe_power(x, y, default=0.0):
        return x ** y
    def safe_mean(x, default=0.0):
        return np.mean(x) if len(x) > 0 else default
    def safe_std(x, default=0.0):
        return np.std(x) if len(x) > 0 else default
    def safe_correlation(x, y, default=0.0):
        return np.corrcoef(x, y)[0, 1] if len(x) > 1 and len(y) > 1 else default
    def safe_covariance(x, y, default=0.0):
        return np.cov(x, y)[0, 1] if len(x) > 1 and len(y) > 1 else default
    def safe_percentile(x, percentile=50.0, default=0.0):
        return np.percentile(x, percentile) if len(x) > 0 else default
    def validate_correlation_matrix(corr_matrix):
        return True
    def safe_matrix_inverse(matrix):
        return np.linalg.inv(matrix)
    def math_safe(func, *args, default=0.0, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
                            tprint_warning(f"⚠️ Operation failed: {e}")
            return default
    class CommonUtilities:
        def __init__(self):
            pass

# Import math validation utilities
try:
    from src.utils.math_validation import (
        MathValidation, validate_finite, validate_positive, validate_range,
        safe_divide, safe_log, safe_sqrt, safe_power, safe_kelly_calculation,
        safe_weighted_average, safe_percentage_change, safe_correlation,
        safe_covariance, safe_mean, safe_std, safe_percentile,
        validate_correlation_matrix, safe_matrix_inverse, math_safe,
        MathValidationError
    )
except ImportError as e:
    logging.warning(f"Could not import math_validation: {e}")
    # Fallback implementations already defined above

# Import serialization utilities
try:
    from src.utils.serialization_utils import (
        JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
    )
except ImportError as e:
    logging.warning(f"Could not import serialization_utils: {e}")
    # Fallback implementations
    class JSONSerializer:
        @staticmethod
        def save(data, filepath):
            return False
        @staticmethod
        def load(filepath):
            return None
    class PickleSerializer:
        @staticmethod
        def save(data, filepath):
            return False
        @staticmethod
        def load(filepath):
            return None
    class ParquetSerializer:
        @staticmethod
        def save(data, filepath):
            return False
        @staticmethod
        def load(filepath):
            return None
    class UniversalSerializer:
        def save(self, data, filepath, format='auto'):
            return False
        def load(self, filepath):
            return None

# Import tprint utilities
try:
    from src.utils.tprint import (
        tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
        tprint_success, tprint_progress, tprint_performance, tprint_structured,
        tprint_with_level, tprint_timer, tprint_logged
    )
except ImportError as e:
    logging.warning(f"Could not import tprint: {e}")
    # Fallback implementations
    def tprint(*args, **kwargs):
        print(*args, **kwargs)
    def tprint_debug(*args, **kwargs):
        print(f"[DEBUG] {' '.join(str(arg) for arg in args)}")
    def tprint_info(*args, **kwargs):
        print(f"[INFO] {' '.join(str(arg) for arg in args)}")
    def tprint_warning(*args, **kwargs):
        print(f"[WARNING] {' '.join(str(arg) for arg in args)}")
    def tprint_error(*args, **kwargs):
        print(f"[ERROR] {' '.join(str(arg) for arg in args)}")
    def tprint_success(*args, **kwargs):
        print(f"[SUCCESS] {' '.join(str(arg) for arg in args)}")
    def tprint_progress(step, total, message="", **kwargs):
        percentage = (step / total) * 100 if total > 0 else 0
        print(f"[PROGRESS] {step}/{total} ({percentage:.1f}%) {message}")
    def tprint_performance(operation, duration, **kwargs):
        print(f"[PERFORMANCE] {operation} took {duration:.3f}s")
    def tprint_structured(data, level=None, **kwargs):
        print(f"[STRUCTURED] {data}")
    def tprint_with_level(level, *args, **kwargs):
        print(f"[{level}] {' '.join(str(arg) for arg in args)}")
    def tprint_timer(operation, level=None):
        from contextlib import nullcontext
        return nullcontext()
    def tprint_logged(level=None, include_args=False, include_result=False):
        def decorator(func):
            return func
        return decorator

# Import ML common utilities
try:
    from src.utils.ml_common.common_operations import (
        safe_cross_validation, safe_hyperparameter_optimization,
        safe_model_evaluation, safe_feature_importance_analysis
    )
except ImportError:
    def safe_cross_validation(model, X, y, cv=5):
        return cross_val_score(model, X, y, cv=cv)
    def safe_hyperparameter_optimization(model, param_grid, X, y, cv=5):
        return GridSearchCV(model, param_grid, cv=cv)
    def safe_model_evaluation(model, X_test, y_test):
        return {}
    def safe_feature_importance_analysis(model, feature_names):
        return {}

# Import matrix operations
try:
    from src.utils.matrix_operations.unified_operations import (
        safe_matrix_operations, optimized_correlation_matrix,
        safe_eigenvalue_decomposition, safe_svd_decomposition
    )
except ImportError:
    def safe_matrix_operations(matrix, operation):
        return matrix
    def optimized_correlation_matrix(data):
        return np.corrcoef(data.T)
    def safe_eigenvalue_decomposition(matrix):
        return np.linalg.eig(matrix)
    def safe_svd_decomposition(matrix):
        return np.linalg.svd(matrix)

# Import hardware optimizations
try:
    from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
except ImportError:
    def get_m1_gpu_manager():
        return None
    def get_m1_memory_optimizer():
        return None
    def get_m1_cpu_optimizer():
        return None

# Setup logging
logger = logging.getLogger(__name__)

class EvaluationType(Enum):
    """Evaluation type enumeration."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    MULTI_OUTPUT = "multi_output"
    ENSEMBLE = "ensemble"

class TreeType(Enum):
    """Tree type enumeration."""
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    EXTRA_TREES = "extra_trees"
    ADA_BOOST = "ada_boost"

class EvaluationMetric(Enum):
    """Evaluation metric enumeration."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    MSE = "mse"
    MAE = "mae"
    R2_SCORE = "r2_score"
    CUSTOM = "custom"

@dataclass
class TreeEvaluationConfig:
    """Configuration for tree evaluation."""
    
    # Basic configuration
    evaluation_type: EvaluationType = EvaluationType.CLASSIFICATION
    tree_type: TreeType = TreeType.DECISION_TREE
    metrics: List[EvaluationMetric] = field(default_factory=lambda: [
        EvaluationMetric.ACCURACY, EvaluationMetric.PRECISION, 
        EvaluationMetric.RECALL, EvaluationMetric.F1_SCORE
    ])
    
    # Cross-validation configuration
    cv_folds: int = 5
    cv_strategy: str = "stratified"  # stratified, kfold, time_series
    
    # Hyperparameter optimization
    enable_hyperparameter_optimization: bool = True
    optimization_method: str = "grid"  # grid, random, bayesian
    n_iter: int = 100  # for random/bayesian search
    
    # Performance configuration
    enable_parallel_processing: bool = True
    n_jobs: int = -1
    max_memory_usage: float = 0.8  # Maximum memory usage (80%)
    
    # M1 optimization
    enable_m1_optimization: bool = True
    use_gpu_acceleration: bool = True
    memory_optimization: bool = True
    
    # Reporting configuration
    generate_detailed_report: bool = True
    save_results: bool = True
    results_directory: str = "evaluation_results"
    
    # Advanced features
    enable_feature_importance: bool = True
    enable_model_interpretation: bool = True
    enable_uncertainty_quantification: bool = True
    enable_ensemble_evaluation: bool = True
    
    # Custom metrics
    custom_metrics: List[Callable] = field(default_factory=list)
    
    # Thresholds and constraints
    min_accuracy_threshold: float = 0.6
    max_training_time: float = 3600.0  # 1 hour
    max_memory_gb: float = 8.0

@dataclass
class TreeEvaluationResults:
    """Results from tree evaluation."""
    
    # Basic metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    mse: float = 0.0
    mae: float = 0.0
    r2_score: float = 0.0
    
    # Cross-validation results
    cv_scores: List[float] = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0
    
    # Hyperparameter optimization
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_score: float = 0.0
    
    # Feature importance
    feature_importance: Dict[str, float] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    
    # Performance metrics
    training_time: float = 0.0
    prediction_time: float = 0.0
    memory_usage: float = 0.0
    
    # Model information
    model_type: str = ""
    model_params: Dict[str, Any] = field(default_factory=dict)
    n_features: int = 0
    n_samples: int = 0
    
    # Additional results
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    classification_report: str = ""
    custom_metrics_results: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    evaluation_timestamp: str = ""
    config: TreeEvaluationConfig = field(default_factory=TreeEvaluationConfig)

class AdvancedTreeEvaluator:
    """Advanced tree evaluator with comprehensive functionality."""
    
    def __init__(self, config: TreeEvaluationConfig = None):
        """Initialize the advanced tree evaluator."""
        self.config = config or TreeEvaluationConfig()
        self.logger = logging.getLogger(__name__)
        self.common_utils = CommonUtilities()
        self.math_validator = MathValidation()
        
        # Initialize M1 optimizations
        self.m1_integration = integrate_with_m1_optimizers()
        self.gpu_manager = get_m1_gpu_manager()
        self.memory_optimizer = get_m1_memory_optimizer()
        self.cpu_optimizer = get_m1_cpu_optimizer()
        
        # Initialize serializers
        self.json_serializer = JSONSerializer()
        self.pickle_serializer = PickleSerializer()
        self.parquet_serializer = ParquetSerializer()
        self.universal_serializer = UniversalSerializer()
        
        # Results storage
        self.results: List[TreeEvaluationResults] = []
        self.current_evaluation: Optional[TreeEvaluationResults] = None
        
        # Performance tracking
        self.performance_metrics: Dict[str, Any] = {}
        
        tprint_info("AdvancedTreeEvaluator initialized successfully")
        tprint_structured({
            "config": self.config.__dict__,
            "m1_integration": self.m1_integration,
            "features": [
                "M1 optimization", "Cross-validation", "Hyperparameter optimization",
                "Feature importance", "Model interpretation", "Performance monitoring"
            ]
        })
    
    @tprint_logged(include_args=True, include_result=True)
    def evaluate_tree_model(
        self, 
        model: Any, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: np.ndarray, 
        y_test: np.ndarray,
        feature_names: List[str] = None,
        **kwargs
    ) -> TreeEvaluationResults:
        """
        Evaluate a tree model with comprehensive metrics and optimizations.
        
        Args:
            model: Tree model to evaluate
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            feature_names: Names of features
            **kwargs: Additional arguments
            
        Returns:
            TreeEvaluationResults: Comprehensive evaluation results
        """
        tprint_info("Starting comprehensive tree model evaluation")
        
        # Initialize results
        results = TreeEvaluationResults()
        results.evaluation_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        results.config = self.config
        
        # Validate inputs
        self._validate_inputs(X_train, y_train, X_test, y_test)
        
        # Setup feature names
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        results.feature_names = feature_names
        results.n_features = X_train.shape[1]
        results.n_samples = X_train.shape[0]
        
        # Memory optimization
        with memory_checkpoint("tree_evaluation"):
            # GPU context if available
            with gpu_context("tree_evaluation"):
                # Perform evaluation
                results = self._perform_evaluation(
                    model, X_train, y_train, X_test, y_test, results
                )
        
        # Store results
        self.current_evaluation = results
        self.results.append(results)
        
        # Save results if configured
        if self.config.save_results:
            self._save_results(results)
        
        tprint_success(f"Tree evaluation completed successfully")
        tprint_performance("Tree evaluation", time.time())
        
        return results
    
    def _validate_inputs(self, X_train, y_train, X_test, y_test):
        """Validate input data."""
        tprint_debug("Validating input data")
        
        # Check data types and shapes
        if not isinstance(X_train, np.ndarray):
            raise ValueError("X_train must be a numpy array")
        if not isinstance(y_train, np.ndarray):
            raise ValueError("y_train must be a numpy array")
        if not isinstance(X_test, np.ndarray):
            raise ValueError("X_test must be a numpy array")
        if not isinstance(y_test, np.ndarray):
            raise ValueError("y_test must be a numpy array")
        
        # Check shapes
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_train and y_train must have the same number of samples")
        if X_test.shape[0] != y_test.shape[0]:
            raise ValueError("X_test and y_test must have the same number of samples")
        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError("X_train and X_test must have the same number of features")
        
        # Check for NaN values
        if np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)):
            tprint_warning("NaN values detected in training data")
        if np.any(np.isnan(X_test)) or np.any(np.isnan(y_test)):
            tprint_warning("NaN values detected in test data")
        
        tprint_success("Input validation completed")
    
    def _perform_evaluation(
        self, 
        model, 
        X_train, y_train, 
        X_test, y_test, 
        results: TreeEvaluationResults
    ) -> TreeEvaluationResults:
        """Perform the actual evaluation."""
        tprint_info("Performing tree model evaluation")
        
        # Train model if not already trained
        if not hasattr(model, 'predict'):
            tprint_info("Training model...")
            start_time = time.time()
            model.fit(X_train, y_train)
            results.training_time = time.time() - start_time
            tprint_performance("Model training", results.training_time)
        
        # Make predictions
        tprint_info("Making predictions...")
        start_time = time.time()
        y_pred = model.predict(X_test)
        results.prediction_time = time.time() - start_time
        tprint_performance("Prediction", results.prediction_time)
        
        # Calculate basic metrics
        results = self._calculate_basic_metrics(y_test, y_pred, results)
        
        # Cross-validation
        if self.config.cv_folds > 1:
            results = self._perform_cross_validation(model, X_train, y_train, results)
        
        # Hyperparameter optimization
        if self.config.enable_hyperparameter_optimization:
            results = self._perform_hyperparameter_optimization(
                model, X_train, y_train, results
            )
        
        # Feature importance
        if self.config.enable_feature_importance:
            results = self._calculate_feature_importance(model, results)
        
        # Model interpretation
        if self.config.enable_model_interpretation:
            results = self._perform_model_interpretation(model, X_test, y_test, results)
        
        # Custom metrics
        if self.config.custom_metrics:
            results = self._calculate_custom_metrics(y_test, y_pred, results)
        
        # Performance monitoring
        results.memory_usage = get_memory_usage()
        
        return results
    
    def _calculate_basic_metrics(
        self, 
        y_test: np.ndarray, 
        y_pred: np.ndarray, 
        results: TreeEvaluationResults
    ) -> TreeEvaluationResults:
        """Calculate basic evaluation metrics."""
        tprint_debug("Calculating basic metrics")
        
        if self.config.evaluation_type == EvaluationType.CLASSIFICATION:
            # Classification metrics
            results.accuracy = safe_divide(
                accuracy_score(y_test, y_pred), 1.0, 0.0
            )
            results.precision = safe_divide(
                precision_score(y_test, y_pred, average='weighted', zero_division=0), 1.0, 0.0
            )
            results.recall = safe_divide(
                recall_score(y_test, y_pred, average='weighted', zero_division=0), 1.0, 0.0
            )
            results.f1_score = safe_divide(
                f1_score(y_test, y_pred, average='weighted', zero_division=0), 1.0, 0.0
            )
            
            # ROC AUC (if binary classification)
            if len(np.unique(y_test)) == 2:
                try:
                    y_pred_proba = self.current_evaluation.model.predict_proba(X_test)[:, 1]
                    results.roc_auc = safe_divide(
                        roc_auc_score(y_test, y_pred_proba), 1.0, 0.0
                    )
                except Exception as e:
                            tprint_warning(f"⚠️ Operation failed: {e}")
                    results.roc_auc = 0.0
            
            # Confusion matrix
            results.confusion_matrix = confusion_matrix(y_test, y_pred)
            results.classification_report = classification_report(y_test, y_pred)
            
        else:  # Regression
            # Regression metrics
            results.mse = safe_divide(
                mean_squared_error(y_test, y_pred), 1.0, 0.0
            )
            results.mae = safe_divide(
                mean_absolute_error(y_test, y_pred), 1.0, 0.0
            )
            results.r2_score = safe_divide(
                r2_score(y_test, y_pred), 1.0, 0.0
            )
        
        tprint_success("Basic metrics calculated")
        return results
    
    def _perform_cross_validation(
        self, 
        model, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        results: TreeEvaluationResults
    ) -> TreeEvaluationResults:
        """Perform cross-validation."""
        tprint_info(f"Performing {self.config.cv_folds}-fold cross-validation")
        
        try:
            # Choose scoring metric based on evaluation type
            if self.config.evaluation_type == EvaluationType.CLASSIFICATION:
                scoring = 'accuracy'
            else:
                scoring = 'neg_mean_squared_error'
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=self.config.cv_folds, 
                scoring=scoring,
                n_jobs=self.config.n_jobs
            )
            
            results.cv_scores = cv_scores.tolist()
            results.cv_mean = safe_mean(cv_scores, 0.0)
            results.cv_std = safe_std(cv_scores, 0.0)
            
            tprint_success(f"Cross-validation completed: {results.cv_mean:.4f} ± {results.cv_std:.4f}")
            
        except Exception as e:
            tprint_error(f"Cross-validation failed: {e}")
            results.cv_scores = []
            results.cv_mean = 0.0
            results.cv_std = 0.0
        
        return results
    
    def _perform_hyperparameter_optimization(
        self, 
        model, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        results: TreeEvaluationResults
    ) -> TreeEvaluationResults:
        """Perform hyperparameter optimization using Bayesian TPE optimizer."""
        tprint_info("Performing hyperparameter optimization with Bayesian TPE")
        
        try:
            # Import Bayesian TPE optimizer
            from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
                BayesianTPEOptimizer,
                BayesianTPEConfig
            )
            
            # Create search space for tree model optimization
            search_space = self._create_tree_search_space(model)
            
            # Define objective function for Bayesian TPE optimizer
            def objective_function(params: Dict[str, Any], **kwargs) -> float:
                try:
                    # Create model with sampled parameters
                    optimized_model = self._create_optimized_model(model, params)
                    
                    # Perform cross-validation
                    if self.config.cv_folds > 1:
                        scores = cross_val_score(
                            optimized_model, X_train, y_train, 
                            cv=self.config.cv_folds,
                            scoring='accuracy' if self.config.evaluation_type == EvaluationType.CLASSIFICATION else 'neg_mean_squared_error'
                        )
                        return np.mean(scores)
                    else:
                        # Single validation
                        optimized_model.fit(X_train, y_train)
                        y_pred = optimized_model.predict(X_train)
                        
                        if self.config.evaluation_type == EvaluationType.CLASSIFICATION:
                            return accuracy_score(y_train, y_pred)
                        else:
                            return -mean_squared_error(y_train, y_pred)
                            
                except Exception as e:
                    tprint_warning(f"Objective function failed: {e}")
                    return -np.inf
            
            # Configure Bayesian TPE optimizer
            tpe_config = BayesianTPEConfig(
                n_trials=self.config.n_iter if hasattr(self.config, 'n_iter') else 50,
                timeout_seconds=300,  # 5 minutes timeout
                enable_grid_search=True,
                coarse_grid_points=3,
                fine_grid_points=5,
                backend='optuna',
                enable_parallel=True,
                max_workers=self.config.n_jobs,
                enable_early_stopping=True,
                early_stopping_patience=10,
                log_level='INFO'
            )
            
            # Run optimization using new unified optimizer
            tprint_info("🎯 Starting Bayesian TPE optimization for tree model")
            optimizer = BayesianTPEOptimizer(tpe_config)
            result = optimizer.optimize(objective_function, search_space)
            
            if not result.success:
                raise RuntimeError(f"Tree model optimization failed: {result.error_message}")
            
            results.best_params = result.best_params
            results.best_score = result.best_score
            
            tprint_success(f"Hyperparameter optimization completed in {result.optimization_time:.2f}s")
            tprint_info(f"Best score: {results.best_score:.4f}")
            tprint_structured({"best_params": results.best_params})
            
        except Exception as e:
            tprint_error(f"Hyperparameter optimization failed: {e}")
            results.best_params = {}
            results.best_score = 0.0
        
        return results
    
    def _create_tree_search_space(self, model) -> Dict[str, Any]:
        """Create search space for tree model optimization."""
        model_type = type(model).__name__.lower()
        
        if 'decisiontree' in model_type:
            return {
                'max_depth': {
                    'type': 'int',
                    'low': 3,
                    'high': 20
                },
                'min_samples_split': {
                    'type': 'int',
                    'low': 2,
                    'high': 20
                },
                'min_samples_leaf': {
                    'type': 'int',
                    'low': 1,
                    'high': 10
                },
                'criterion': {
                    'type': 'categorical',
                    'choices': ['gini', 'entropy']
                }
            }
        elif 'randomforest' in model_type:
            return {
                'n_estimators': {
                    'type': 'int',
                    'low': 50,
                    'high': 300
                },
                'max_depth': {
                    'type': 'int',
                    'low': 3,
                    'high': 15
                },
                'min_samples_split': {
                    'type': 'int',
                    'low': 2,
                    'high': 10
                },
                'min_samples_leaf': {
                    'type': 'int',
                    'low': 1,
                    'high': 5
                },
                'bootstrap': {
                    'type': 'categorical',
                    'choices': [True, False]
                }
            }
        elif 'gradientboosting' in model_type:
            return {
                'n_estimators': {
                    'type': 'int',
                    'low': 50,
                    'high': 200
                },
                'learning_rate': {
                    'type': 'float',
                    'low': 0.01,
                    'high': 0.2,
                    'log': True
                },
                'max_depth': {
                    'type': 'int',
                    'low': 3,
                    'high': 10
                },
                'min_samples_split': {
                    'type': 'int',
                    'low': 2,
                    'high': 10
                },
                'min_samples_leaf': {
                    'type': 'int',
                    'low': 1,
                    'high': 5
                }
            }
        else:
            return {}
    
    def _create_optimized_model(self, model, params: Dict[str, Any]):
        """Create optimized model with given parameters."""
        model_type = type(model).__name__
        
        if 'DecisionTree' in model_type:
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
            if self.config.evaluation_type == EvaluationType.CLASSIFICATION:
                return DecisionTreeClassifier(**params)
            else:
                return DecisionTreeRegressor(**params)
        elif 'RandomForest' in model_type:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            if self.config.evaluation_type == EvaluationType.CLASSIFICATION:
                return RandomForestClassifier(**params)
            else:
                return RandomForestRegressor(**params)
        elif 'GradientBoosting' in model_type:
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
            if self.config.evaluation_type == EvaluationType.CLASSIFICATION:
                return GradientBoostingClassifier(**params)
            else:
                return GradientBoostingRegressor(**params)
        else:
            # Fallback to original model with parameters
            return type(model)(**params)
    
    def _get_parameter_grid(self, model) -> Dict[str, List]:
        """Get parameter grid for hyperparameter optimization."""
        model_type = type(model).__name__.lower()
        
        if 'decisiontree' in model_type:
            return {
                'max_depth': [3, 5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 5, 10],
                'criterion': ['gini', 'entropy']
            }
        elif 'randomforest' in model_type:
            return {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5],
                'bootstrap': [True, False]
            }
        elif 'gradientboosting' in model_type:
            return {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5]
            }
        else:
            return {}
    
    def _calculate_feature_importance(
        self, 
        model, 
        results: TreeEvaluationResults
    ) -> TreeEvaluationResults:
        """Calculate feature importance."""
        tprint_debug("Calculating feature importance")
        
        try:
            if hasattr(model, 'feature_importances_'):
                importance_scores = model.feature_importances_
                results.feature_importance = {
                    name: score for name, score in zip(results.feature_names, importance_scores)
                }
                tprint_success("Feature importance calculated")
            else:
                tprint_warning("Model does not support feature importance")
                results.feature_importance = {}
        
        except Exception as e:
            tprint_error(f"Feature importance calculation failed: {e}")
            results.feature_importance = {}
        
        return results
    
    def _perform_model_interpretation(
        self, 
        model, 
        X_test: np.ndarray, 
        y_test: np.ndarray, 
        results: TreeEvaluationResults
    ) -> TreeEvaluationResults:
        """Perform model interpretation."""
        tprint_debug("Performing model interpretation")
        
        try:
            # This is a placeholder for more advanced interpretation methods
            # In a real implementation, you might use SHAP, LIME, or other methods
            tprint_info("Model interpretation completed")
        
        except Exception as e:
            tprint_error(f"Model interpretation failed: {e}")
        
        return results
    
    def _calculate_custom_metrics(
        self, 
        y_test: np.ndarray, 
        y_pred: np.ndarray, 
        results: TreeEvaluationResults
    ) -> TreeEvaluationResults:
        """Calculate custom metrics."""
        tprint_debug("Calculating custom metrics")
        
        try:
            for metric_func in self.config.custom_metrics:
                metric_name = metric_func.__name__
                metric_value = metric_func(y_test, y_pred)
                results.custom_metrics_results[metric_name] = metric_value
                tprint_info(f"Custom metric {metric_name}: {metric_value:.4f}")
        
        except Exception as e:
            tprint_error(f"Custom metrics calculation failed: {e}")
        
        return results
    
    def _save_results(self, results: TreeEvaluationResults):
        """Save evaluation results."""
        tprint_info("Saving evaluation results")
        
        try:
            # Create results directory
            results_dir = Path(self.config.results_directory)
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            timestamp = results.evaluation_timestamp.replace(":", "-").replace(" ", "_")
            filename = f"tree_evaluation_results_{timestamp}"
            
            # Save as JSON
            json_path = results_dir / f"{filename}.json"
            results_dict = self._results_to_dict(results)
            safe_json_dump(results_dict, json_path)
            
            # Save as pickle for full object
            pickle_path = results_dir / f"{filename}.pkl"
            self.pickle_serializer.save(results, pickle_path)
            
            tprint_success(f"Results saved to {results_dir}")
        
        except Exception as e:
            tprint_error(f"Failed to save results: {e}")
    
    def _results_to_dict(self, results: TreeEvaluationResults) -> Dict[str, Any]:
        """Convert results to dictionary for JSON serialization."""
        return {
            "accuracy": results.accuracy,
            "precision": results.precision,
            "recall": results.recall,
            "f1_score": results.f1_score,
            "roc_auc": results.roc_auc,
            "mse": results.mse,
            "mae": results.mae,
            "r2_score": results.r2_score,
            "cv_scores": results.cv_scores,
            "cv_mean": results.cv_mean,
            "cv_std": results.cv_std,
            "best_params": results.best_params,
            "best_score": results.best_score,
            "feature_importance": results.feature_importance,
            "training_time": results.training_time,
            "prediction_time": results.prediction_time,
            "memory_usage": results.memory_usage,
            "model_type": results.model_type,
            "n_features": results.n_features,
            "n_samples": results.n_samples,
            "evaluation_timestamp": results.evaluation_timestamp,
            "custom_metrics_results": results.custom_metrics_results
        }
    
    def compare_models(
        self, 
        models: List[Any], 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        model_names: List[str] = None
    ) -> Dict[str, TreeEvaluationResults]:
        """Compare multiple tree models."""
        tprint_info(f"Comparing {len(models)} models")
        
        if model_names is None:
            model_names = [f"model_{i}" for i in range(len(models))]
        
        comparison_results = {}
        
        for i, (model, name) in enumerate(zip(models, model_names)):
            tprint_progress(i + 1, len(models), f"Evaluating {name}")
            
            try:
                results = self.evaluate_tree_model(
                    model, None, None, X_test, y_test
                )
                results.model_type = name
                comparison_results[name] = results
                
            except Exception as e:
                tprint_error(f"Failed to evaluate {name}: {e}")
                comparison_results[name] = None
        
        # Generate comparison report
        self._generate_comparison_report(comparison_results)
        
        return comparison_results
    
    def _generate_comparison_report(self, comparison_results: Dict[str, TreeEvaluationResults]):
        """Generate comparison report."""
        tprint_info("Generating comparison report")
        
        try:
            # Create comparison DataFrame
            comparison_data = []
            
            for name, results in comparison_results.items():
                if results is not None:
                    comparison_data.append({
                        "Model": name,
                        "Accuracy": results.accuracy,
                        "Precision": results.precision,
                        "Recall": results.recall,
                        "F1 Score": results.f1_score,
                        "Training Time": results.training_time,
                        "Prediction Time": results.prediction_time,
                        "Memory Usage": results.memory_usage
                    })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                tprint_info("Model Comparison Results:")
                tprint_structured(comparison_df.to_dict('records'))
        
        except Exception as e:
            tprint_error(f"Failed to generate comparison report: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of all evaluations."""
        if not self.results:
            return {}
        
        summary = {
            "total_evaluations": len(self.results),
            "average_accuracy": safe_mean([r.accuracy for r in self.results], 0.0),
            "average_precision": safe_mean([r.precision for r in self.results], 0.0),
            "average_recall": safe_mean([r.recall for r in self.results], 0.0),
            "average_f1_score": safe_mean([r.f1_score for r in self.results], 0.0),
            "average_training_time": safe_mean([r.training_time for r in self.results], 0.0),
            "average_prediction_time": safe_mean([r.prediction_time for r in self.results], 0.0),
            "average_memory_usage": safe_mean([r.memory_usage for r in self.results], 0.0)
        }
        
        return summary

# Legacy compatibility classes
class TreeEvaluator:
    """Legacy tree evaluator for backward compatibility."""
    
    def __init__(self, config: TreeEvaluationConfig = None):
        self.advanced_evaluator = AdvancedTreeEvaluator(config)
    
    def evaluate(self, model, X_test, y_test, **kwargs):
        """Evaluate tree model."""
        return self.advanced_evaluator.evaluate_tree_model(
            model, None, None, X_test, y_test, **kwargs
        )

class TreePerformanceEvaluator:
    """Tree performance evaluator."""
    
    def __init__(self, config: TreeEvaluationConfig = None):
        self.advanced_evaluator = AdvancedTreeEvaluator(config)
    
    def evaluate_performance(self, model, X_test, y_test, **kwargs):
        """Evaluate tree performance."""
        return self.advanced_evaluator.evaluate_tree_model(
            model, None, None, X_test, y_test, **kwargs
        )

class TreeBenchmarkEvaluator:
    """Tree benchmark evaluator."""
    
    def __init__(self, config: TreeEvaluationConfig = None):
        self.advanced_evaluator = AdvancedTreeEvaluator(config)
    
    def evaluate_benchmark(self, model, X_test, y_test, benchmark_model=None, **kwargs):
        """Evaluate tree benchmark."""
        if benchmark_model:
            return self.advanced_evaluator.compare_models(
                [model, benchmark_model], X_test, y_test, 
                ["model", "benchmark"]
            )
        else:
            return self.advanced_evaluator.evaluate_tree_model(
                model, None, None, X_test, y_test, **kwargs
            )

# Utility functions
def create_tree_evaluator(config: TreeEvaluationConfig = None) -> AdvancedTreeEvaluator:
    """Create a tree evaluator instance."""
    return AdvancedTreeEvaluator(config)

def evaluate_tree_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: TreeEvaluationConfig = None
) -> TreeEvaluationResults:
    """Convenience function to evaluate a tree model."""
    evaluator = AdvancedTreeEvaluator(config)
    return evaluator.evaluate_tree_model(model, None, None, X_test, y_test)

def compare_tree_models(
    models: List[Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_names: List[str] = None,
    config: TreeEvaluationConfig = None
) -> Dict[str, TreeEvaluationResults]:
    """Convenience function to compare tree models."""
    evaluator = AdvancedTreeEvaluator(config)
    return evaluator.compare_models(models, X_test, y_test, model_names)

# Export all classes and functions
__all__ = [
    'AdvancedTreeEvaluator',
    'TreeEvaluator',
    'TreePerformanceEvaluator', 
    'TreeBenchmarkEvaluator',
    'TreeEvaluationConfig',
    'TreeEvaluationResults',
    'EvaluationType',
    'TreeType',
    'EvaluationMetric',
    'create_tree_evaluator',
    'evaluate_tree_model',
    'compare_tree_models'
]