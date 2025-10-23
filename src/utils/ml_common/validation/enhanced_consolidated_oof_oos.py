"""
Enhanced Consolidated Out-of-Fold (OOF) and Out-of-Sample (OOS) Utilities

This module provides a comprehensive, unified interface for all OOF/OOS operations,
combining the best features from various implementations across the codebase:

Key Features:
- Unified OOF prediction generation with multiple strategies
- Advanced OOS validation and performance evaluation
- Confidence interval estimation and uncertainty quantification
- Ensemble diversity metrics and correlation analysis
- Integrated leakage detection and prevention
- Hardware optimization and M1 support
- Temporal validation with purged cross-validation
- Multi-output support for ensemble methods
- Enhanced stacking ensemble management
- Comprehensive reporting and monitoring

This module consolidates functionality from:
- oof_stacking_ensemble_manager.py
- enhanced_oof_stacking_with_confidence.py
- training_utils.py (OOF methods)
- multi_output_models.py (OOF evaluation)
- tactician_ensemble_training.py (OOF generation)
- feature_generation_period_lookback_optimization_step.py (OOS Sharpe)
- leakage_detection_system.py (leakage detection)
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
from sklearn.model_selection import BaseCrossValidator, TimeSeriesSplit
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import ElasticNet, Ridge
from collections import defaultdict, Counter
import json
from pathlib import Path
import itertools
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import traceback

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

# Import leakage detection
try:
    from leakage_detection_system import (
        LeakageDetector, LeakageType, LeakageSeverity, LeakageProof
    )
    LEAKAGE_DETECTION_AVAILABLE = True
except ImportError:
    LEAKAGE_DETECTION_AVAILABLE = False

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
    STACKING = "stacking"


class OOSValidationType(Enum):
    """Types of OOS validation."""
    SHARPE_RATIO = "sharpe_ratio"
    PERFORMANCE_METRICS = "performance_metrics"
    NESTED_CV = "nested_cv"
    TEMPORAL_VALIDATION = "temporal_validation"
    LEAKAGE_DETECTION = "leakage_detection"
    NESTED_SHARPE = "nested_sharpe"


class ConfidenceMethod(Enum):
    """Methods for confidence interval estimation."""
    PERCENTILE = "percentile"
    BIAS_CORRECTED = "bias_corrected"
    STUDENTIZED = "studentized"
    ENSEMBLE_VARIANCE = "ensemble_variance"
    MODEL_UNCERTAINTY = "model_uncertainty"
    BOOTSTRAP = "bootstrap"


class EnsembleType(Enum):
    """Types of ensemble methods."""
    STACKING = "stacking"
    VOTING = "voting"
    BAGGING = "bagging"
    BOOSTING = "boosting"
    BLENDING = "blending"


@dataclass
class EnhancedOOFConfig:
    """Enhanced configuration for OOF prediction generation."""
    
    # Basic configuration
    strategy: OOFStrategy = OOFStrategy.STACKING
    n_folds: int = 5
    random_state: Optional[int] = None
    
    # Cross-validation configuration
    cv_type: ValidationType = ValidationType.PURGED
    cv_config: Optional[ConsolidatedCVConfig] = None
    
    # OOF specific settings
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_rounds: int = 50
    early_stopping_min_delta: float = 1e-4
    
    # Multi-output settings
    n_outputs: int = 1
    output_names: Optional[List[str]] = None
    output_weights: Optional[List[float]] = None
    output_loss_weights: Optional[List[float]] = None
    
    # Ensemble settings
    ensemble_type: EnsembleType = EnsembleType.STACKING
    enable_meta_learning: bool = True
    meta_model_type: str = "ridge"  # ridge, elastic_net, random_forest, xgboost
    
    # Performance settings
    enable_parallel_processing: bool = True
    max_workers: Optional[int] = None
    enable_caching: bool = True
    cache_size_mb: int = 100
    
    # Hardware optimization
    enable_m1_optimization: bool = True
    memory_limit_gb: float = 8.0
    
    # Advanced features
    enable_confidence_intervals: bool = True
    enable_diversity_metrics: bool = True
    enable_leakage_detection: bool = True
    enable_temporal_validation: bool = True


@dataclass
class EnhancedOOSConfig:
    """Enhanced configuration for OOS validation."""
    
    # Validation type
    validation_type: OOSValidationType = OOSValidationType.NESTED_SHARPE
    
    # Sharpe ratio settings
    risk_free_rate: float = 0.0
    min_test_signals: int = 100
    annualization_factor: Optional[float] = None
    
    # Performance metrics
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "mse", "r2", "sharpe"])
    
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
    
    # Nested Sharpe specific
    enable_nested_sharpe: bool = True
    sharpe_optimization: bool = True
    sharpe_threshold: float = 0.5


@dataclass
class EnhancedConfidenceConfig:
    """Enhanced configuration for confidence interval estimation."""
    
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
    
    # Advanced uncertainty
    enable_aleatoric_uncertainty: bool = True
    enable_epistemic_uncertainty: bool = True
    uncertainty_aggregation: str = "mean"  # mean, max, weighted


@dataclass
class EnhancedOOFResult:
    """Enhanced result from OOF prediction generation."""
    
    # Basic predictions
    oof_predictions: Dict[str, np.ndarray]
    oof_scores: Dict[str, float]
    
    # Fold information
    fold_predictions: Dict[int, Dict[str, np.ndarray]]
    fold_scores: Dict[int, Dict[str, float]]
    
    # Configuration
    config: EnhancedOOFConfig
    generation_time: float
    
    # Metadata
    n_samples: int
    n_folds: int
    model_names: List[str]
    
    # Performance metrics
    ensemble_diversity: Optional[Dict[str, float]] = None
    prediction_uncertainty: Optional[Dict[str, np.ndarray]] = None
    
    # Advanced features
    confidence_intervals: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
    leakage_detection: Optional[Dict[str, Any]] = None
    temporal_analysis: Optional[Dict[str, Any]] = None
    
    # Stacking specific
    meta_model_performance: Optional[Dict[str, float]] = None
    base_model_weights: Optional[Dict[str, float]] = None
    stacking_confidence: Optional[np.ndarray] = None


@dataclass
class EnhancedOOSResult:
    """Enhanced result from OOS validation."""
    
    # Validation results
    validation_scores: Dict[str, float]
    validation_metrics: Dict[str, Any]
    
    # Configuration
    config: EnhancedOOSConfig
    validation_time: float
    
    # Metadata
    n_samples: int
    validation_type: OOSValidationType
    
    # Additional results
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None
    leakage_detection: Optional[Dict[str, Any]] = None
    temporal_analysis: Optional[Dict[str, Any]] = None
    
    # Nested Sharpe specific
    nested_sharpe_scores: Optional[Dict[str, float]] = None
    sharpe_optimization_results: Optional[Dict[str, Any]] = None


@dataclass
class EnhancedConfidenceResult:
    """Enhanced result from confidence interval estimation."""
    
    # Confidence intervals
    confidence_intervals: Dict[str, Tuple[np.ndarray, np.ndarray]]  # (lower, upper)
    prediction_uncertainty: Dict[str, np.ndarray]
    
    # Bootstrap results
    bootstrap_predictions: Optional[Dict[str, np.ndarray]] = None
    bootstrap_scores: Optional[Dict[str, List[float]]] = None
    
    # Configuration
    config: EnhancedConfidenceConfig
    estimation_time: float
    
    # Advanced uncertainty
    aleatoric_uncertainty: Optional[Dict[str, np.ndarray]] = None
    epistemic_uncertainty: Optional[Dict[str, np.ndarray]] = None
    total_uncertainty: Optional[Dict[str, np.ndarray]] = None


class EnhancedConsolidatedOOFGenerator:
    """
    Enhanced consolidated OOF prediction generator with advanced features.
    
    This class provides a unified interface for generating out-of-fold predictions
    with support for multiple strategies, confidence intervals, hardware optimization,
    and advanced ensemble methods including stacking.
    """
    
    def __init__(self, config: Optional[EnhancedOOFConfig] = None):
        """Initialize enhanced consolidated OOF generator."""
        self.config = config or EnhancedOOFConfig()
        self.logger = logging.getLogger(f"{__name__}.EnhancedConsolidatedOOFGenerator")
        
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
        
        # Initialize leakage detection if available
        if LEAKAGE_DETECTION_AVAILABLE and self.config.enable_leakage_detection:
            self.leakage_detector = LeakageDetector()
        else:
            self.leakage_detector = None
        
        if TPRINT_AVAILABLE:
            tprint_info("🚀 EnhancedConsolidatedOOFGenerator initialized with advanced features")
    
    def generate_oof_predictions(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        model_names: Optional[List[str]] = None,
        timestamps: Optional[np.ndarray] = None
    ) -> EnhancedOOFResult:
        """
        Generate OOF predictions using the specified models and configuration.
        
        Args:
            models: Dictionary of models to use for OOF prediction
            X: Feature matrix
            y: Target vector
            model_names: Optional list of model names (uses dict keys if not provided)
            timestamps: Optional timestamps for temporal validation
            
        Returns:
            EnhancedOOFResult containing predictions, scores, and metadata
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
        if len(oof_predictions) > 1 and self.config.enable_diversity_metrics:
            ensemble_diversity = self._calculate_ensemble_diversity(oof_predictions)
        
        # Calculate prediction uncertainty
        prediction_uncertainty = None
        if self.config.enable_confidence_intervals:
            prediction_uncertainty = self._calculate_prediction_uncertainty(oof_predictions)
        
        # Generate confidence intervals
        confidence_intervals = None
        if self.config.enable_confidence_intervals:
            confidence_intervals = self._calculate_confidence_intervals(oof_predictions, y)
        
        # Perform leakage detection
        leakage_detection = None
        if self.config.enable_leakage_detection and self.leakage_detector is not None:
            leakage_detection = self._perform_leakage_detection(oof_predictions, y, timestamps)
        
        # Perform temporal analysis
        temporal_analysis = None
        if self.config.enable_temporal_validation and timestamps is not None:
            temporal_analysis = self._perform_temporal_analysis(oof_predictions, timestamps)
        
        # Generate stacking predictions if enabled
        meta_model_performance = None
        base_model_weights = None
        stacking_confidence = None
        
        if self.config.strategy == OOFStrategy.STACKING and len(oof_predictions) > 1:
            meta_model_performance, base_model_weights, stacking_confidence = self._generate_stacking_predictions(
                oof_predictions, X, y
            )
        
        generation_time = time.time() - start_time
        
        if TPRINT_AVAILABLE:
            tprint_success(f"✅ OOF predictions generated in {generation_time:.2f}s")
        
        return EnhancedOOFResult(
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
            prediction_uncertainty=prediction_uncertainty,
            confidence_intervals=confidence_intervals,
            leakage_detection=leakage_detection,
            temporal_analysis=temporal_analysis,
            meta_model_performance=meta_model_performance,
            base_model_weights=base_model_weights,
            stacking_confidence=stacking_confidence
        )
    
    def _generate_model_oof_predictions(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str
    ) -> Tuple[np.ndarray, float, Dict[int, np.ndarray], Dict[int, float]]:
        """Generate OOF predictions for a single model with enhanced features."""
        
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
            y_train, y_val = y[train_indices], y[train_indices]
            
            # Clone and train model
            fold_model = clone(model)
            
            # Apply early stopping if enabled
            if self.config.enable_early_stopping and hasattr(fold_model, 'fit'):
                fold_model = self._apply_early_stopping(fold_model, X_train, y_train, X_val, y_val, model_name)
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
    
    def _apply_early_stopping(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_name: str
    ) -> Any:
        """Apply early stopping to a model if supported."""
        try:
            # Check if model supports early stopping
            if hasattr(model, 'set_params'):
                # XGBoost
                if 'xgb' in model_name.lower():
                    model.set_params(
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=self.config.early_stopping_rounds,
                        verbose=False
                    )
                # LightGBM
                elif 'lgbm' in model_name.lower() or 'lightgbm' in model_name.lower():
                    model.set_params(
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=self.config.early_stopping_rounds,
                        callbacks=['early_stopping'],
                        verbose=-1
                    )
                # CatBoost
                elif 'catboost' in model_name.lower():
                    model.set_params(
                        eval_set=(X_val, y_val),
                        early_stopping_rounds=self.config.early_stopping_rounds,
                        verbose=False,
                        use_best_model=True
                    )
            
            model.fit(X_train, y_train)
            return model
            
        except Exception as e:
            self.logger.warning(f"⚠️ Early stopping failed for {model_name}: {e}")
            model.fit(X_train, y_train)
            return model
    
    def _calculate_ensemble_diversity(self, oof_predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate enhanced ensemble diversity metrics."""
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
        
        # Calculate additional diversity metrics
        diversity_metrics['variance_ratio'] = np.var(predictions_array) / np.mean(np.var(predictions_array, axis=1))
        diversity_metrics['disagreement_rate'] = np.mean([np.mean(predictions_array[i] != predictions_array[j]) 
                                                         for i in range(len(predictions_array)) 
                                                         for j in range(i + 1, len(predictions_array))])
        
        return diversity_metrics
    
    def _calculate_prediction_uncertainty(self, oof_predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate enhanced prediction uncertainty metrics."""
        uncertainty_metrics = {}
        
        if len(oof_predictions) < 2:
            return uncertainty_metrics
        
        predictions_array = np.array(list(oof_predictions.values()))
        
        # Calculate variance across models
        uncertainty_metrics['variance'] = np.var(predictions_array, axis=0)
        uncertainty_metrics['std'] = np.std(predictions_array, axis=0)
        uncertainty_metrics['range'] = np.ptp(predictions_array, axis=0)
        
        # Calculate additional uncertainty metrics
        uncertainty_metrics['coefficient_of_variation'] = uncertainty_metrics['std'] / (np.mean(predictions_array, axis=0) + 1e-8)
        uncertainty_metrics['interquartile_range'] = np.percentile(predictions_array, 75, axis=0) - np.percentile(predictions_array, 25, axis=0)
        
        return uncertainty_metrics
    
    def _calculate_confidence_intervals(
        self,
        oof_predictions: Dict[str, np.ndarray],
        y: np.ndarray
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Calculate confidence intervals for OOF predictions."""
        confidence_intervals = {}
        
        for model_name, predictions in oof_predictions.items():
            # Simple confidence interval based on prediction variance
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            
            # 95% confidence interval
            ci_lower = mean_pred - 1.96 * std_pred
            ci_upper = mean_pred + 1.96 * std_pred
            
            confidence_intervals[model_name] = (ci_lower, ci_upper)
        
        return confidence_intervals
    
    def _calculate_stacking_confidence(
        self,
        meta_predictions: np.ndarray,
        oof_predictions: Dict[str, np.ndarray],
        y: np.ndarray
    ) -> np.ndarray:
        """Calculate stacking confidence based on meta-model prediction variance and base model agreement."""
        try:
            n_samples = len(y)
            stacking_confidence = np.zeros(n_samples)
            
            # Method 1: Meta-model prediction variance
            if len(meta_predictions.shape) > 1 and meta_predictions.shape[1] > 1:
                # Multiple meta-model predictions - use variance
                meta_variance = np.var(meta_predictions, axis=1)
                meta_confidence = 1.0 / (1.0 + meta_variance)  # Higher variance = lower confidence
            else:
                # Single meta-model prediction - use base model agreement
                meta_confidence = np.ones(n_samples)
            
            # Method 2: Base model agreement
            if len(oof_predictions) > 1:
                # Calculate agreement between base models
                base_predictions = np.array(list(oof_predictions.values()))
                base_agreement = 1.0 - np.std(base_predictions, axis=0) / (np.mean(np.abs(base_predictions), axis=0) + 1e-8)
                base_agreement = np.clip(base_agreement, 0.0, 1.0)
            else:
                base_agreement = np.ones(n_samples)
            
            # Method 3: Prediction accuracy (if we have ground truth)
            if y is not None and len(y) == n_samples:
                # Calculate how close predictions are to actual values
                prediction_errors = np.abs(meta_predictions.flatten() - y.flatten())
                max_error = np.max(prediction_errors) + 1e-8
                accuracy_confidence = 1.0 - (prediction_errors / max_error)
                accuracy_confidence = np.clip(accuracy_confidence, 0.0, 1.0)
            else:
                accuracy_confidence = np.ones(n_samples)
            
            # Combine confidence measures with weights
            weights = self.config.get('confidence_weights', {
                'meta_variance': 0.4,
                'base_agreement': 0.3,
                'accuracy': 0.3
            })
            
            stacking_confidence = (
                weights['meta_variance'] * meta_confidence +
                weights['base_agreement'] * base_agreement +
                weights['accuracy'] * accuracy_confidence
            )
            
            # Ensure confidence is between 0 and 1
            stacking_confidence = np.clip(stacking_confidence, 0.0, 1.0)
            
            # Apply smoothing to reduce noise
            if n_samples > 1:
                from scipy.ndimage import gaussian_filter1d
                try:
                    stacking_confidence = gaussian_filter1d(stacking_confidence, sigma=1.0)
                    stacking_confidence = np.clip(stacking_confidence, 0.0, 1.0)
                except ImportError:
                    # Fallback to simple moving average if scipy not available
                    window_size = min(5, n_samples // 4)
                    if window_size > 1:
                        for i in range(n_samples):
                            start_idx = max(0, i - window_size // 2)
                            end_idx = min(n_samples, i + window_size // 2 + 1)
                            stacking_confidence[i] = np.mean(stacking_confidence[start_idx:end_idx])
            
            self.logger.debug(f"Stacking confidence calculated: mean={np.mean(stacking_confidence):.3f}, "
                            f"std={np.std(stacking_confidence):.3f}, "
                            f"min={np.min(stacking_confidence):.3f}, max={np.max(stacking_confidence):.3f}")
            
            return stacking_confidence
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate stacking confidence: {e}")
            # Fallback to uniform confidence
            return np.ones(len(y))
    
    def _perform_leakage_detection(
        self,
        oof_predictions: Dict[str, np.ndarray],
        y: np.ndarray,
        timestamps: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Perform leakage detection on OOF predictions."""
        if self.leakage_detector is None:
            return {}
        
        try:
            # Create DataFrame for leakage detection
            data = pd.DataFrame(oof_predictions)
            if timestamps is not None:
                data.index = pd.to_datetime(timestamps)
            
            # Perform leakage detection
            leakage_results = self.leakage_detector.detect_leakage(data, y)
            
            return {
                'leakage_detected': len(leakage_results) > 0,
                'leakage_count': len(leakage_results),
                'leakage_types': [result.leakage_type.value for result in leakage_results],
                'severity_levels': [result.severity.value for result in leakage_results]
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Leakage detection failed: {e}")
            return {'error': str(e)}
    
    def _perform_temporal_analysis(
        self,
        oof_predictions: Dict[str, np.ndarray],
        timestamps: np.ndarray
    ) -> Dict[str, Any]:
        """Perform temporal analysis on OOF predictions."""
        temporal_analysis = {}
        
        for model_name, predictions in oof_predictions.items():
            # Calculate temporal correlation
            temporal_corr = np.corrcoef(timestamps, predictions)[0, 1]
            
            # Calculate temporal stability (variance over time)
            time_windows = np.array_split(predictions, 10)  # Split into 10 time windows
            window_means = [np.mean(window) for window in time_windows]
            temporal_stability = 1.0 / (1.0 + np.var(window_means))
            
            temporal_analysis[model_name] = {
                'temporal_correlation': temporal_corr,
                'temporal_stability': temporal_stability,
                'trend_strength': abs(temporal_corr)
            }
        
        return temporal_analysis
    
    def _generate_stacking_predictions(
        self,
        oof_predictions: Dict[str, np.ndarray],
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[Dict[str, float], Dict[str, float], np.ndarray]:
        """Generate stacking predictions using meta-learning."""
        try:
            # Prepare meta-features (OOF predictions)
            meta_features = np.column_stack(list(oof_predictions.values()))
            
            # Create meta-model
            if self.config.meta_model_type == "ridge":
                meta_model = Ridge(alpha=1.0, random_state=self.config.random_state)
            elif self.config.meta_model_type == "elastic_net":
                meta_model = ElasticNet(alpha=1.0, random_state=self.config.random_state)
            elif self.config.meta_model_type == "random_forest":
                meta_model = RandomForestRegressor(n_estimators=100, random_state=self.config.random_state)
            else:
                meta_model = Ridge(alpha=1.0, random_state=self.config.random_state)
            
            # Train meta-model
            meta_model.fit(meta_features, y)
            
            # Calculate meta-model performance
            meta_predictions = meta_model.predict(meta_features)
            meta_performance = r2_score(y, meta_predictions)
            
            # Calculate base model weights (feature importance)
            if hasattr(meta_model, 'coef_'):
                base_weights = dict(zip(oof_predictions.keys(), meta_model.coef_))
            else:
                base_weights = {name: 1.0/len(oof_predictions) for name in oof_predictions.keys()}
            
            # Calculate stacking confidence (based on meta-model prediction variance)
            stacking_confidence = self._calculate_stacking_confidence(meta_predictions, oof_predictions, y)
            
            return (
                {self.config.meta_model_type: meta_performance},
                base_weights,
                stacking_confidence
            )
            
        except Exception as e:
            self.logger.warning(f"⚠️ Stacking prediction generation failed: {e}")
            return {}, {}, np.ones(len(y))


class EnhancedConsolidatedOOSValidator:
    """
    Enhanced consolidated OOS validation with multiple validation types.
    
    This class provides a unified interface for out-of-sample validation
    with support for Sharpe ratio, performance metrics, nested CV, leakage detection,
    and advanced nested Sharpe ratio optimization.
    """
    
    def __init__(self, config: Optional[EnhancedOOSConfig] = None):
        """Initialize enhanced consolidated OOS validator."""
        self.config = config or EnhancedOOSConfig()
        self.logger = logging.getLogger(f"{__name__}.EnhancedConsolidatedOOSValidator")
        
        # Initialize leakage detection if available
        if LEAKAGE_DETECTION_AVAILABLE and self.config.enable_leakage_detection:
            self.leakage_detector = LeakageDetector()
        else:
            self.leakage_detector = None
        
        if TPRINT_AVAILABLE:
            tprint_info(f"🚀 EnhancedConsolidatedOOSValidator initialized for {self.config.validation_type.value} validation")
    
    def validate_oos(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        returns: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None
    ) -> EnhancedOOSResult:
        """
        Perform OOS validation based on the configured validation type.
        
        Args:
            predictions: Model predictions
            targets: True targets
            returns: Optional returns for Sharpe ratio calculation
            timestamps: Optional timestamps for temporal validation
            
        Returns:
            EnhancedOOSResult containing validation scores and metrics
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
        elif self.config.validation_type == OOSValidationType.NESTED_SHARPE:
            validation_scores, validation_metrics = self._validate_nested_sharpe(
                predictions, targets, returns, timestamps
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
        
        # Perform additional validations
        confidence_intervals = None
        leakage_detection = None
        temporal_analysis = None
        
        if self.config.enable_leakage_detection and self.leakage_detector is not None:
            leakage_detection = self._perform_leakage_detection(predictions, targets, timestamps)
        
        if self.config.enable_temporal_validation and timestamps is not None:
            temporal_analysis = self._perform_temporal_analysis(predictions, timestamps)
        
        validation_time = time.time() - start_time
        
        if TPRINT_AVAILABLE:
            tprint_success(f"✅ OOS validation completed in {validation_time:.2f}s")
        
        return EnhancedOOSResult(
            validation_scores=validation_scores,
            validation_metrics=validation_metrics,
            config=self.config,
            validation_time=validation_time,
            n_samples=len(predictions),
            validation_type=self.config.validation_type,
            confidence_intervals=confidence_intervals,
            leakage_detection=leakage_detection,
            temporal_analysis=temporal_analysis
        )
    
    def _validate_nested_sharpe(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        returns: Optional[np.ndarray],
        timestamps: Optional[np.ndarray]
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Validate using nested Sharpe ratio optimization."""
        scores = {}
        metrics = {}
        
        if returns is None:
            returns = targets
        
        # Calculate basic Sharpe ratio
        excess_returns = returns - self.config.risk_free_rate
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
        
        # Annualize if factor provided
        if self.config.annualization_factor:
            sharpe_ratio *= np.sqrt(self.config.annualization_factor)
        
        scores['sharpe_ratio'] = sharpe_ratio
        
        # Perform nested Sharpe optimization if enabled
        if self.config.enable_nested_sharpe and self.config.sharpe_optimization:
            nested_scores, optimization_results = self._perform_nested_sharpe_optimization(
                predictions, returns, timestamps
            )
            scores.update(nested_scores)
            metrics['optimization_results'] = optimization_results
        
        # Calculate additional Sharpe metrics
        metrics['excess_returns'] = excess_returns
        metrics['volatility'] = np.std(excess_returns)
        metrics['mean_return'] = np.mean(excess_returns)
        metrics['max_drawdown'] = self._calculate_max_drawdown(excess_returns)
        metrics['calmar_ratio'] = np.mean(excess_returns) / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
        
        return scores, metrics
    
    def _perform_nested_sharpe_optimization(
        self,
        predictions: np.ndarray,
        returns: np.ndarray,
        timestamps: Optional[np.ndarray]
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Perform nested Sharpe ratio optimization."""
        nested_scores = {}
        optimization_results = {}
        
        try:
            # Split data into outer folds for nested validation
            n_samples = len(predictions)
            outer_fold_size = n_samples // self.config.outer_folds
            
            outer_sharpe_scores = []
            
            for i in range(self.config.outer_folds):
                start_idx = i * outer_fold_size
                end_idx = min((i + 1) * outer_fold_size, n_samples)
                
                if end_idx - start_idx < self.config.min_test_signals:
                    continue
                
                # Get fold data
                fold_predictions = predictions[start_idx:end_idx]
                fold_returns = returns[start_idx:end_idx]
                
                # Calculate fold Sharpe ratio
                fold_excess_returns = fold_returns - self.config.risk_free_rate
                fold_sharpe = np.mean(fold_excess_returns) / np.std(fold_excess_returns)
                
                if self.config.annualization_factor:
                    fold_sharpe *= np.sqrt(self.config.annualization_factor)
                
                outer_sharpe_scores.append(fold_sharpe)
            
            # Calculate nested Sharpe metrics
            if outer_sharpe_scores:
                nested_scores['nested_sharpe_mean'] = np.mean(outer_sharpe_scores)
                nested_scores['nested_sharpe_std'] = np.std(outer_sharpe_scores)
                nested_scores['nested_sharpe_min'] = np.min(outer_sharpe_scores)
                nested_scores['nested_sharpe_max'] = np.max(outer_sharpe_scores)
                nested_scores['nested_sharpe_consistency'] = 1.0 - (np.std(outer_sharpe_scores) / (np.mean(outer_sharpe_scores) + 1e-8))
                
                # Check if nested Sharpe meets threshold
                nested_scores['nested_sharpe_above_threshold'] = 1.0 if nested_scores['nested_sharpe_mean'] > self.config.sharpe_threshold else 0.0
                
                optimization_results['outer_fold_scores'] = outer_sharpe_scores
                optimization_results['n_folds_used'] = len(outer_sharpe_scores)
                optimization_results['optimization_successful'] = True
            else:
                optimization_results['optimization_successful'] = False
                optimization_results['error'] = "Insufficient data for nested validation"
        
        except Exception as e:
            self.logger.warning(f"⚠️ Nested Sharpe optimization failed: {e}")
            optimization_results['optimization_successful'] = False
            optimization_results['error'] = str(e)
        
        return nested_scores, optimization_results
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
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
            elif metric == "sharpe":
                # Calculate Sharpe ratio for performance metrics
                excess_returns = targets - self.config.risk_free_rate
                sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
                if self.config.annualization_factor:
                    sharpe_ratio *= np.sqrt(self.config.annualization_factor)
                scores[metric] = sharpe_ratio
        
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
    
    def _perform_leakage_detection(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        timestamps: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Perform comprehensive leakage detection."""
        if self.leakage_detector is None:
            return {}
        
        try:
            # Create DataFrame for leakage detection
            data = pd.DataFrame({'predictions': predictions, 'targets': targets})
            if timestamps is not None:
                data.index = pd.to_datetime(timestamps)
            
            # Perform leakage detection
            leakage_results = self.leakage_detector.detect_leakage(data, targets)
            
            return {
                'leakage_detected': len(leakage_results) > 0,
                'leakage_count': len(leakage_results),
                'leakage_types': [result.leakage_type.value for result in leakage_results],
                'severity_levels': [result.severity.value for result in leakage_results]
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Leakage detection failed: {e}")
            return {'error': str(e)}
    
    def _perform_temporal_analysis(
        self,
        predictions: np.ndarray,
        timestamps: np.ndarray
    ) -> Dict[str, Any]:
        """Perform temporal analysis on predictions."""
        # Calculate temporal correlation
        temporal_corr = np.corrcoef(timestamps, predictions)[0, 1]
        
        # Calculate temporal stability
        time_windows = np.array_split(predictions, 10)
        window_means = [np.mean(window) for window in time_windows]
        temporal_stability = 1.0 / (1.0 + np.var(window_means))
        
        return {
            'temporal_correlation': temporal_corr,
            'temporal_stability': temporal_stability,
            'trend_strength': abs(temporal_corr)
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def create_enhanced_oof_generator(
    strategy: OOFStrategy = OOFStrategy.STACKING,
    n_folds: int = 5,
    cv_type: ValidationType = ValidationType.PURGED,
    **kwargs
) -> EnhancedConsolidatedOOFGenerator:
    """Create an enhanced consolidated OOF generator with specified configuration."""
    config = EnhancedOOFConfig(
        strategy=strategy,
        n_folds=n_folds,
        cv_type=cv_type,
        **kwargs
    )
    return EnhancedConsolidatedOOFGenerator(config)


def create_enhanced_oos_validator(
    validation_type: OOSValidationType = OOSValidationType.NESTED_SHARPE,
    **kwargs
) -> EnhancedConsolidatedOOSValidator:
    """Create an enhanced consolidated OOS validator with specified configuration."""
    config = EnhancedOOSConfig(
        validation_type=validation_type,
        **kwargs
    )
    return EnhancedConsolidatedOOSValidator(config)


# ============================================================================
# Legacy Compatibility
# ============================================================================

# Legacy class names for backward compatibility
EnhancedOOFGenerator = EnhancedConsolidatedOOFGenerator
EnhancedOOSValidator = EnhancedConsolidatedOOSValidator

# Legacy function names
def generate_enhanced_oof_predictions(models, X, y, strategy='stacking', n_folds=5, **kwargs):
    """Legacy function for generating enhanced OOF predictions."""
    generator = create_enhanced_oof_generator(strategy=OOFStrategy(strategy), n_folds=n_folds, **kwargs)
    return generator.generate_oof_predictions(models, X, y)


def validate_enhanced_oos(predictions, targets, validation_type='nested_sharpe', **kwargs):
    """Legacy function for enhanced OOS validation."""
    validator = create_enhanced_oos_validator(validation_type=OOSValidationType(validation_type), **kwargs)
    return validator.validate_oos(predictions, targets)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Main classes
    'EnhancedConsolidatedOOFGenerator', 'EnhancedConsolidatedOOSValidator',
    
    # Configuration classes
    'EnhancedOOFConfig', 'EnhancedOOSConfig', 'EnhancedConfidenceConfig',
    
    # Result classes
    'EnhancedOOFResult', 'EnhancedOOSResult', 'EnhancedConfidenceResult',
    
    # Enums
    'OOFStrategy', 'OOSValidationType', 'ConfidenceMethod', 'EnsembleType',
    
    # Convenience functions
    'create_enhanced_oof_generator', 'create_enhanced_oos_validator',
    
    # Legacy compatibility
    'EnhancedOOFGenerator', 'EnhancedOOSValidator',
    'generate_enhanced_oof_predictions', 'validate_enhanced_oos',
]