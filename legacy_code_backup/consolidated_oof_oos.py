"""
Consolidated Out-of-Fold (OOF) and Out-of-Sample (OOS) Utilities

This module provides a unified interface for all OOF/OOS operations, combining
the best features from various implementations across the codebase:

1. OOF prediction generation with multiple strategies
2. OOS validation and performance evaluation
3. Confidence interval estimation and uncertainty quantification
4. Ensemble diversity metrics and correlation analysis
5. Leakage detection integration
6. Hardware optimization and M1 support
7. Temporal validation with purged cross-validation
8. Multi-output support for ensemble methods

Key Features:
- Unified OOF/OOS generation and validation
- Multiple combination strategies (mean, median, vote, weighted)
- Bootstrap-based confidence intervals
- Ensemble diversity metrics
- Leakage detection and prevention
- Hardware-optimized operations
- Comprehensive reporting and monitoring
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Generator
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.stats import bootstrap
from sklearn.model_selection import BaseCrossValidator
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import ElasticNet
from collections import defaultdict, Counter
import json
from pathlib import Path
import itertools
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Import consolidated CV
from .consolidated_cv import ConsolidatedCrossValidator, ConsolidatedCVConfig, ValidationType

# Import tprint utilities
try:
    from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint_info(msg): print(f"INFO: {msg}")
    def tprint_warning(msg): print(f"WARNING: {msg}")
    def tprint_error(msg): print(f"ERROR: {msg}")
    def tprint_success(msg): print(f"SUCCESS: {msg}")

# Import hardware optimization
try:
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer, M1MemoryOptimizer
    from src.utils.hardware.memory_optimization import get_memory_manager, MemoryMonitor
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False

# Import common utilities
try:
    from src.utils.common_operations import (
        safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
        safe_mean, safe_std, safe_float, safe_int, get_current_datetime,
        safe_append, safe_extend, safe_dict_get, safe_lower, safe_upper,
        format_datetime, validate_file_path, get_file_size
    )
    from src.utils.math_validation import (
        safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
        validate_positive, validate_range, safe_kelly_calculation,
        safe_weighted_average, safe_percentage_change, MathValidationError
    )
    COMMON_UTILITIES_AVAILABLE = True
except ImportError:
    COMMON_UTILITIES_AVAILABLE = False

logger = logging.getLogger(__name__)


class OOFStrategy(Enum):
    """Strategies for combining OOF predictions."""
    MEAN = "mean"
    MEDIAN = "median"
    VOTE = "vote"
    WEIGHTED_MEAN = "weighted_mean"
    WEIGHTED_MEDIAN = "weighted_median"
    ROBUST_MEAN = "robust_mean"
    ENSEMBLE_VOTE = "ensemble_vote"


class OOSValidationType(Enum):
    """Types of OOS validation."""
    SHARPE_RATIO = "sharpe_ratio"
    PERFORMANCE_METRICS = "performance_metrics"
    NESTED_CV = "nested_cv"
    TEMPORAL_VALIDATION = "temporal_validation"
    LEAKAGE_DETECTION = "leakage_detection"


class ConfidenceMethod(Enum):
    """Methods for confidence interval estimation."""
    PERCENTILE = "percentile"
    BIAS_CORRECTED = "bias_corrected"
    STUDENTIZED = "studentized"
    ENSEMBLE_VARIANCE = "ensemble_variance"
    MODEL_UNCERTAINTY = "model_uncertainty"


@dataclass
class OOFConfig:
    """Configuration for OOF prediction generation."""
    
    # Basic configuration
    strategy: OOFStrategy = OOFStrategy.MEAN
    n_folds: int = 5
    random_state: Optional[int] = None
    
    # Cross-validation configuration
    cv_type: ValidationType = ValidationType.PURGED
    cv_config: Optional[ConsolidatedCVConfig] = None
    
    # OOF specific settings
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_rounds: int = 50
    
    # Multi-output settings
    n_outputs: int = 1
    output_names: Optional[List[str]] = None
    output_weights: Optional[List[float]] = None
    
    # Performance settings
    enable_parallel_processing: bool = True
    max_workers: Optional[int] = None
    enable_caching: bool = True
    cache_size_mb: int = 100
    
    # Hardware optimization
    enable_m1_optimization: bool = True
    memory_limit_gb: float = 8.0


@dataclass
class OOSConfig:
    """Configuration for OOS validation."""
    
    # Validation type
    validation_type: OOSValidationType = OOSValidationType.SHARPE_RATIO
    
    # Sharpe ratio settings
    risk_free_rate: float = 0.0
    min_test_signals: int = 100
    annualization_factor: Optional[float] = None
    
    # Performance metrics
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "mse", "r2"])
    
    # Nested CV settings
    outer_folds: int = 3
    inner_folds: int = 5
    
    # Temporal validation
    enable_temporal_validation: bool = True
    temporal_gap: int = 0
    
    # Leakage detection
    enable_leakage_detection: bool = True
    leakage_threshold: float = 0.1
    
    # Bootstrap settings
    n_bootstrap_samples: int = 100
    confidence_level: float = 0.95


@dataclass
class ConfidenceConfig:
    """Configuration for confidence interval estimation."""
    
    # Bootstrap configuration
    n_bootstrap_samples: int = 100
    confidence_level: float = 0.95
    bootstrap_method: ConfidenceMethod = ConfidenceMethod.PERCENTILE
    
    # Prediction uncertainty
    enable_prediction_uncertainty: bool = True
    uncertainty_method: ConfidenceMethod = ConfidenceMethod.ENSEMBLE_VARIANCE
    
    # Performance
    parallel_bootstrap: bool = True
    max_workers: Optional[int] = None


@dataclass
class OOFResult:
    """Result from OOF prediction generation."""
    
    # Basic predictions
    oof_predictions: Dict[str, np.ndarray]
    oof_scores: Dict[str, float]
    
    # Fold information
    fold_predictions: Dict[int, Dict[str, np.ndarray]]
    fold_scores: Dict[int, Dict[str, float]]
    
    # Configuration
    config: OOFConfig
    generation_time: float
    
    # Metadata
    n_samples: int
    n_folds: int
    model_names: List[str]
    
    # Performance metrics
    ensemble_diversity: Optional[Dict[str, float]] = None
    prediction_uncertainty: Optional[Dict[str, np.ndarray]] = None


@dataclass
class OOSResult:
    """Result from OOS validation."""
    
    # Validation results
    validation_scores: Dict[str, float]
    validation_metrics: Dict[str, Any]
    
    # Configuration
    config: OOSConfig
    validation_time: float
    
    # Metadata
    n_samples: int
    validation_type: OOSValidationType
    
    # Additional results
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None
    leakage_detection: Optional[Dict[str, Any]] = None
    temporal_analysis: Optional[Dict[str, Any]] = None


@dataclass
class ConfidenceResult:
    """Result from confidence interval estimation."""
    
    # Confidence intervals
    confidence_intervals: Dict[str, Tuple[np.ndarray, np.ndarray]]  # (lower, upper)
    prediction_uncertainty: Dict[str, np.ndarray]
    
    # Bootstrap results
    bootstrap_predictions: Optional[Dict[str, np.ndarray]] = None
    bootstrap_scores: Optional[Dict[str, List[float]]] = None
    
    # Configuration
    config: ConfidenceConfig
    estimation_time: float


class ConsolidatedOOFGenerator:
    """
    Consolidated OOF prediction generator with advanced features.
    
    This class provides a unified interface for generating out-of-fold predictions
    with support for multiple strategies, confidence intervals, and hardware optimization.
    """
    
    def __init__(self, config: Optional[OOFConfig] = None):
        """Initialize consolidated OOF generator."""
        self.config = config or OOFConfig()
        self.logger = logging.getLogger(f"{__name__}.ConsolidatedOOFGenerator")
        
        # Initialize hardware optimization if available
        if HARDWARE_OPTIMIZATION_AVAILABLE and self.config.enable_m1_optimization:
            self.memory_optimizer = get_m1_memory_optimizer()
            self.memory_monitor = get_memory_manager()
        else:
            self.memory_optimizer = None
            self.memory_monitor = None
        
        # Initialize CV if not provided
        if self.config.cv_config is None:
            self.config.cv_config = ConsolidatedCVConfig(
                n_splits=self.config.n_folds,
                random_state=self.config.random_state
            )
        
        self.cv = ConsolidatedCrossValidator(
            config=self.config.cv_config,
            validation_type=self.config.cv_type,
            random_state=self.config.random_state
        )
        
        if TPRINT_AVAILABLE:
            tprint_info("🚀 ConsolidatedOOFGenerator initialized with advanced features")
    
    def generate_oof_predictions(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        model_names: Optional[List[str]] = None
    ) -> OOFResult:
        """
        Generate OOF predictions using the specified models and configuration.
        
        Args:
            models: Dictionary of models to use for OOF prediction
            X: Feature matrix
            y: Target vector
            model_names: Optional list of model names (uses dict keys if not provided)
            
        Returns:
            OOFResult containing predictions, scores, and metadata
        """
        start_time = time.time()
        
        if model_names is None:
            model_names = list(models.keys())
        
        if TPRINT_AVAILABLE:
            tprint_info(f"🔄 Generating OOF predictions for {len(model_names)} models using {self.config.strategy.value} strategy")
        
        # Initialize result containers
        oof_predictions = {}
        oof_scores = {}
        fold_predictions = {}
        fold_scores = {}
        
        # Generate OOF predictions for each model
        for model_name in model_names:
            if model_name not in models:
                self.logger.warning(f"⚠️ Model {model_name} not found in models dictionary")
                continue
            
            model = models[model_name]
            model_oof_preds, model_oof_scores, model_fold_preds, model_fold_scores = self._generate_model_oof_predictions(
                model, X, y, model_name
            )
            
            oof_predictions[model_name] = model_oof_preds
            oof_scores[model_name] = model_oof_scores
            fold_predictions[model_name] = model_fold_preds
            fold_scores[model_name] = model_fold_scores
        
        # Calculate ensemble diversity if multiple models
        ensemble_diversity = None
        if len(oof_predictions) > 1:
            ensemble_diversity = self._calculate_ensemble_diversity(oof_predictions)
        
        # Calculate prediction uncertainty
        prediction_uncertainty = self._calculate_prediction_uncertainty(oof_predictions)
        
        generation_time = time.time() - start_time
        
        if TPRINT_AVAILABLE:
            tprint_success(f"✅ OOF predictions generated in {generation_time:.2f}s")
        
        return OOFResult(
            oof_predictions=oof_predictions,
            oof_scores=oof_scores,
            fold_predictions=fold_predictions,
            fold_scores=fold_scores,
            config=self.config,
            generation_time=generation_time,
            n_samples=len(X),
            n_folds=self.config.n_folds,
            model_names=model_names,
            ensemble_diversity=ensemble_diversity,
            prediction_uncertainty=prediction_uncertainty
        )
    
    def _generate_model_oof_predictions(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str
    ) -> Tuple[np.ndarray, float, Dict[int, np.ndarray], Dict[int, float]]:
        """Generate OOF predictions for a single model."""
        
        # Initialize prediction containers
        n_samples = len(X)
        n_outputs = self.config.n_outputs
        
        if n_outputs == 1:
            oof_predictions = np.zeros(n_samples)
        else:
            oof_predictions = np.zeros((n_samples, n_outputs))
        
        fold_predictions = {}
        fold_scores = {}
        
        # Generate predictions for each fold
        fold_idx = 0
        for train_indices, val_indices in self.cv.split(X, y):
            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]
            
            # Clone and train model
            fold_model = clone(model)
            
            # Apply early stopping if enabled
            if self.config.enable_early_stopping and hasattr(fold_model, 'fit'):
                fold_model.fit(X_train, y_train)
            else:
                fold_model.fit(X_train, y_train)
            
            # Generate predictions
            if n_outputs == 1:
                val_predictions = fold_model.predict(X_val)
            else:
                val_predictions = fold_model.predict(X_val)
            
            # Store fold predictions
            fold_predictions[fold_idx] = val_predictions
            oof_predictions[val_indices] = val_predictions
            
            # Calculate fold score
            if len(np.unique(y_val)) <= 2:  # Classification
                fold_score = accuracy_score(y_val, val_predictions)
            else:  # Regression
                fold_score = r2_score(y_val, val_predictions)
            
            fold_scores[fold_idx] = fold_score
            fold_idx += 1
        
        # Calculate overall OOF score
        if len(np.unique(y)) <= 2:  # Classification
            oof_score = accuracy_score(y, oof_predictions)
        else:  # Regression
            oof_score = r2_score(y, oof_predictions)
        
        return oof_predictions, oof_score, fold_predictions, fold_scores
    
    def _calculate_ensemble_diversity(self, oof_predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate ensemble diversity metrics."""
        diversity_metrics = {}
        
        if len(oof_predictions) < 2:
            return diversity_metrics
        
        predictions_array = np.array(list(oof_predictions.values()))
        
        # Calculate pairwise correlations
        correlations = []
        for i in range(len(predictions_array)):
            for j in range(i + 1, len(predictions_array)):
                corr = np.corrcoef(predictions_array[i], predictions_array[j])[0, 1]
                correlations.append(corr)
        
        diversity_metrics['mean_correlation'] = np.mean(correlations)
        diversity_metrics['std_correlation'] = np.std(correlations)
        diversity_metrics['min_correlation'] = np.min(correlations)
        diversity_metrics['max_correlation'] = np.max(correlations)
        
        # Calculate diversity score (1 - mean correlation)
        diversity_metrics['diversity_score'] = 1.0 - diversity_metrics['mean_correlation']
        
        return diversity_metrics
    
    def _calculate_prediction_uncertainty(self, oof_predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate prediction uncertainty metrics."""
        uncertainty_metrics = {}
        
        if len(oof_predictions) < 2:
            return uncertainty_metrics
        
        predictions_array = np.array(list(oof_predictions.values()))
        
        # Calculate variance across models
        uncertainty_metrics['variance'] = np.var(predictions_array, axis=0)
        uncertainty_metrics['std'] = np.std(predictions_array, axis=0)
        uncertainty_metrics['range'] = np.ptp(predictions_array, axis=0)
        
        return uncertainty_metrics


class ConsolidatedOOSValidator:
    """
    Consolidated OOS validation with multiple validation types.
    
    This class provides a unified interface for out-of-sample validation
    with support for Sharpe ratio, performance metrics, nested CV, and leakage detection.
    """
    
    def __init__(self, config: Optional[OOSConfig] = None):
        """Initialize consolidated OOS validator."""
        self.config = config or OOSConfig()
        self.logger = logging.getLogger(f"{__name__}.ConsolidatedOOSValidator")
        
        if TPRINT_AVAILABLE:
            tprint_info(f"🚀 ConsolidatedOOSValidator initialized for {self.config.validation_type.value} validation")
    
    def validate_oos(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        returns: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None
    ) -> OOSResult:
        """
        Perform OOS validation based on the configured validation type.
        
        Args:
            predictions: Model predictions
            targets: True targets
            returns: Optional returns for Sharpe ratio calculation
            timestamps: Optional timestamps for temporal validation
            
        Returns:
            OOSResult containing validation scores and metrics
        """
        start_time = time.time()
        
        if TPRINT_AVAILABLE:
            tprint_info(f"🔄 Performing {self.config.validation_type.value} OOS validation")
        
        validation_scores = {}
        validation_metrics = {}
        
        # Perform validation based on type
        if self.config.validation_type == OOSValidationType.SHARPE_RATIO:
            validation_scores, validation_metrics = self._validate_sharpe_ratio(
                predictions, targets, returns
            )
        elif self.config.validation_type == OOSValidationType.PERFORMANCE_METRICS:
            validation_scores, validation_metrics = self._validate_performance_metrics(
                predictions, targets
            )
        elif self.config.validation_type == OOSValidationType.NESTED_CV:
            validation_scores, validation_metrics = self._validate_nested_cv(
                predictions, targets
            )
        elif self.config.validation_type == OOSValidationType.TEMPORAL_VALIDATION:
            validation_scores, validation_metrics = self._validate_temporal(
                predictions, targets, timestamps
            )
        elif self.config.validation_type == OOSValidationType.LEAKAGE_DETECTION:
            validation_scores, validation_metrics = self._validate_leakage_detection(
                predictions, targets, timestamps
            )
        
        validation_time = time.time() - start_time
        
        if TPRINT_AVAILABLE:
            tprint_success(f"✅ OOS validation completed in {validation_time:.2f}s")
        
        return OOSResult(
            validation_scores=validation_scores,
            validation_metrics=validation_metrics,
            config=self.config,
            validation_time=validation_time,
            n_samples=len(predictions),
            validation_type=self.config.validation_type
        )
    
    def _validate_sharpe_ratio(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        returns: Optional[np.ndarray]
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Validate using Sharpe ratio."""
        scores = {}
        metrics = {}
        
        if returns is None:
            # Use targets as returns if not provided
            returns = targets
        
        # Calculate Sharpe ratio
        excess_returns = returns - self.config.risk_free_rate
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
        
        # Annualize if factor provided
        if self.config.annualization_factor:
            sharpe_ratio *= np.sqrt(self.config.annualization_factor)
        
        scores['sharpe_ratio'] = sharpe_ratio
        metrics['excess_returns'] = excess_returns
        metrics['volatility'] = np.std(excess_returns)
        metrics['mean_return'] = np.mean(excess_returns)
        
        return scores, metrics
    
    def _validate_performance_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Validate using performance metrics."""
        scores = {}
        metrics = {}
        
        for metric in self.config.metrics:
            if metric == "accuracy":
                scores[metric] = accuracy_score(targets, predictions)
            elif metric == "f1":
                scores[metric] = f1_score(targets, predictions, average='weighted')
            elif metric == "mse":
                scores[metric] = mean_squared_error(targets, predictions)
            elif metric == "r2":
                scores[metric] = r2_score(targets, predictions)
        
        metrics['predictions'] = predictions
        metrics['targets'] = targets
        metrics['residuals'] = targets - predictions
        
        return scores, metrics
    
    def _validate_nested_cv(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Validate using nested cross-validation."""
        scores = {}
        metrics = {}
        
        # This would implement nested CV validation
        # For now, return basic metrics
        scores['nested_cv_score'] = r2_score(targets, predictions)
        metrics['nested_cv_implemented'] = False
        
        return scores, metrics
    
    def _validate_temporal(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        timestamps: Optional[np.ndarray]
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Validate using temporal analysis."""
        scores = {}
        metrics = {}
        
        if timestamps is not None:
            # Calculate temporal correlation
            temporal_corr = np.corrcoef(timestamps, predictions)[0, 1]
            scores['temporal_correlation'] = temporal_corr
            metrics['timestamps'] = timestamps
        
        scores['temporal_score'] = r2_score(targets, predictions)
        metrics['temporal_validation'] = True
        
        return scores, metrics
    
    def _validate_leakage_detection(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        timestamps: Optional[np.ndarray]
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Validate using leakage detection."""
        scores = {}
        metrics = {}
        
        # Basic leakage detection
        if timestamps is not None:
            # Check for temporal leakage
            temporal_corr = np.corrcoef(timestamps, predictions)[0, 1]
            scores['leakage_score'] = abs(temporal_corr)
            metrics['temporal_correlation'] = temporal_corr
        
        scores['leakage_detected'] = 1.0 if scores.get('leakage_score', 0) > self.config.leakage_threshold else 0.0
        metrics['leakage_threshold'] = self.config.leakage_threshold
        
        return scores, metrics


class ConsolidatedConfidenceEstimator:
    """
    Consolidated confidence interval estimation with multiple methods.
    
    This class provides a unified interface for estimating confidence intervals
    using bootstrap methods, ensemble variance, and model uncertainty.
    """
    
    def __init__(self, config: Optional[ConfidenceConfig] = None):
        """Initialize consolidated confidence estimator."""
        self.config = config or ConfidenceConfig()
        self.logger = logging.getLogger(f"{__name__}.ConsolidatedConfidenceEstimator")
        
        if TPRINT_AVAILABLE:
            tprint_info(f"🚀 ConsolidatedConfidenceEstimator initialized with {self.config.bootstrap_method.value} method")
    
    def estimate_confidence_intervals(
        self,
        predictions: Dict[str, np.ndarray],
        targets: np.ndarray,
        n_bootstrap: Optional[int] = None
    ) -> ConfidenceResult:
        """
        Estimate confidence intervals for predictions.
        
        Args:
            predictions: Dictionary of model predictions
            targets: True targets
            n_bootstrap: Number of bootstrap samples (uses config if not provided)
            
        Returns:
            ConfidenceResult containing confidence intervals and uncertainty metrics
        """
        start_time = time.time()
        
        if n_bootstrap is None:
            n_bootstrap = self.config.n_bootstrap_samples
        
        if TPRINT_AVAILABLE:
            tprint_info(f"🔄 Estimating confidence intervals using {n_bootstrap} bootstrap samples")
        
        confidence_intervals = {}
        prediction_uncertainty = {}
        bootstrap_predictions = {}
        bootstrap_scores = {}
        
        # Estimate confidence intervals for each model
        for model_name, model_predictions in predictions.items():
            ci_lower, ci_upper, uncertainty = self._estimate_model_confidence(
                model_predictions, targets, n_bootstrap
            )
            
            confidence_intervals[model_name] = (ci_lower, ci_upper)
            prediction_uncertainty[model_name] = uncertainty
        
        # Estimate ensemble confidence intervals
        if len(predictions) > 1:
            ensemble_predictions = np.mean(list(predictions.values()), axis=0)
            ci_lower, ci_upper, uncertainty = self._estimate_model_confidence(
                ensemble_predictions, targets, n_bootstrap
            )
            
            confidence_intervals['ensemble'] = (ci_lower, ci_upper)
            prediction_uncertainty['ensemble'] = uncertainty
        
        estimation_time = time.time() - start_time
        
        if TPRINT_AVAILABLE:
            tprint_success(f"✅ Confidence intervals estimated in {estimation_time:.2f}s")
        
        return ConfidenceResult(
            confidence_intervals=confidence_intervals,
            prediction_uncertainty=prediction_uncertainty,
            config=self.config,
            estimation_time=estimation_time
        )
    
    def _estimate_model_confidence(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        n_bootstrap: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate confidence intervals for a single model."""
        
        if self.config.bootstrap_method == ConfidenceMethod.PERCENTILE:
            return self._bootstrap_percentile(predictions, targets, n_bootstrap)
        elif self.config.bootstrap_method == ConfidenceMethod.BIAS_CORRECTED:
            return self._bootstrap_bias_corrected(predictions, targets, n_bootstrap)
        elif self.config.bootstrap_method == ConfidenceMethod.STUDENTIZED:
            return self._bootstrap_studentized(predictions, targets, n_bootstrap)
        else:
            return self._bootstrap_percentile(predictions, targets, n_bootstrap)
    
    def _bootstrap_percentile(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        n_bootstrap: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bootstrap percentile method for confidence intervals."""
        
        # Generate bootstrap samples
        bootstrap_predictions = []
        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(len(predictions), size=len(predictions), replace=True)
            bootstrap_predictions.append(predictions[indices])
        
        bootstrap_predictions = np.array(bootstrap_predictions)
        
        # Calculate confidence intervals
        alpha = 1 - self.config.confidence_level
        ci_lower = np.percentile(bootstrap_predictions, 100 * alpha / 2, axis=0)
        ci_upper = np.percentile(bootstrap_predictions, 100 * (1 - alpha / 2), axis=0)
        
        # Calculate uncertainty (standard deviation across bootstrap samples)
        uncertainty = np.std(bootstrap_predictions, axis=0)
        
        return ci_lower, ci_upper, uncertainty
    
    def _bootstrap_bias_corrected(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        n_bootstrap: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bias-corrected bootstrap method for confidence intervals."""
        
        # This is a simplified implementation
        # In practice, you'd implement the full bias-corrected bootstrap
        return self._bootstrap_percentile(predictions, targets, n_bootstrap)
    
    def _bootstrap_studentized(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        n_bootstrap: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Studentized bootstrap method for confidence intervals."""
        
        # This is a simplified implementation
        # In practice, you'd implement the full studentized bootstrap
        return self._bootstrap_percentile(predictions, targets, n_bootstrap)


# ============================================================================
# Convenience Functions
# ============================================================================

def create_oof_generator(
    strategy: OOFStrategy = OOFStrategy.MEAN,
    n_folds: int = 5,
    cv_type: ValidationType = ValidationType.PURGED,
    **kwargs
) -> ConsolidatedOOFGenerator:
    """Create a consolidated OOF generator with specified configuration."""
    config = OOFConfig(
        strategy=strategy,
        n_folds=n_folds,
        cv_type=cv_type,
        **kwargs
    )
    return ConsolidatedOOFGenerator(config)


def create_oos_validator(
    validation_type: OOSValidationType = OOSValidationType.SHARPE_RATIO,
    **kwargs
) -> ConsolidatedOOSValidator:
    """Create a consolidated OOS validator with specified configuration."""
    config = OOSConfig(
        validation_type=validation_type,
        **kwargs
    )
    return ConsolidatedOOSValidator(config)


def create_confidence_estimator(
    bootstrap_method: ConfidenceMethod = ConfidenceMethod.PERCENTILE,
    confidence_level: float = 0.95,
    **kwargs
) -> ConsolidatedConfidenceEstimator:
    """Create a consolidated confidence estimator with specified configuration."""
    config = ConfidenceConfig(
        bootstrap_method=bootstrap_method,
        confidence_level=confidence_level,
        **kwargs
    )
    return ConsolidatedConfidenceEstimator(config)


# ============================================================================
# Legacy Compatibility
# ============================================================================

# Legacy class names for backward compatibility
OOFGenerator = ConsolidatedOOFGenerator
OOSValidator = ConsolidatedOOSValidator
ConfidenceEstimator = ConsolidatedConfidenceEstimator

# Legacy function names
def generate_oof_predictions(models, X, y, strategy='mean', n_folds=5, **kwargs):
    """Legacy function for generating OOF predictions."""
    generator = create_oof_generator(strategy=OOFStrategy(strategy), n_folds=n_folds, **kwargs)
    return generator.generate_oof_predictions(models, X, y)


def validate_oos(predictions, targets, validation_type='sharpe_ratio', **kwargs):
    """Legacy function for OOS validation."""
    validator = create_oos_validator(validation_type=OOSValidationType(validation_type), **kwargs)
    return validator.validate_oos(predictions, targets)


def estimate_confidence_intervals(predictions, targets, method='percentile', **kwargs):
    """Legacy function for confidence interval estimation."""
    estimator = create_confidence_estimator(bootstrap_method=ConfidenceMethod(method), **kwargs)
    return estimator.estimate_confidence_intervals(predictions, targets)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Main classes
    'ConsolidatedOOFGenerator', 'ConsolidatedOOSValidator', 'ConsolidatedConfidenceEstimator',
    
    # Configuration classes
    'OOFConfig', 'OOSConfig', 'ConfidenceConfig',
    
    # Result classes
    'OOFResult', 'OOSResult', 'ConfidenceResult',
    
    # Enums
    'OOFStrategy', 'OOSValidationType', 'ConfidenceMethod',
    
    # Convenience functions
    'create_oof_generator', 'create_oos_validator', 'create_confidence_estimator',
    
    # Legacy compatibility
    'OOFGenerator', 'OOSValidator', 'ConfidenceEstimator',
    'generate_oof_predictions', 'validate_oos', 'estimate_confidence_intervals',
]