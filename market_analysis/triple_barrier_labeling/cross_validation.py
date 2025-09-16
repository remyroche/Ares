"""
Cross-Validation for Triple Barrier Labels

This module provides comprehensive cross-validation functionality for triple barrier
labels, including temporal cross-validation, purged cross-validation, and regime-aware
validation to ensure label quality and prevent overfitting.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
import warnings
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

# Import common utilities
from src.utils.common_operations import (
    safe_divide, safe_log, safe_sqrt, safe_power,
    validate_finite, validate_positive, validate_range,
    safe_dataframe_operation, validate_dataframe_columns
)
from src.utils.common_utilities import CommonUtilities
from src.utils.math_validation import MathValidation

# Import ML common utilities
from src.utils.ml_common.validation.cv_utils import TemporalCrossValidator, PurgedKFold
from src.utils.ml_common.evaluation.evaluation_utils import EvaluationUtils

# Setup logging
logger = logging.getLogger(__name__)

class CVMethod(Enum):
    """Cross-validation methods."""
    TEMPORAL_CV = "temporal_cv"
    PURGED_CV = "purged_cv"
    TIME_SERIES_CV = "time_series_cv"
    REGIME_AWARE_CV = "regime_aware_cv"
    WALK_FORWARD_CV = "walk_forward_cv"

class ValidationMetric(Enum):
    """Validation metrics."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    SHARPE_RATIO = "sharpe_ratio"
    PROFIT_FACTOR = "profit_factor"
    MAX_DRAWDOWN = "max_drawdown"

@dataclass
class CVConfig:
    """Configuration for cross-validation."""
    # CV method settings
    method: CVMethod = CVMethod.TEMPORAL_CV
    n_splits: int = 5
    purged_pct: float = 0.01
    embargo_pct: float = 0.01
    
    # Regime-aware settings
    regime_aware: bool = False
    regime_column: str = "regime"
    min_samples_per_regime: int = 50
    
    # Walk-forward settings
    initial_train_size: int = 1000
    step_size: int = 100
    
    # Model settings
    models: List[str] = field(default_factory=lambda: ["random_forest", "logistic_regression"])
    random_state: int = 42
    
    # Performance settings
    parallel_processing: bool = True
    max_workers: int = 4
    
    # Quality thresholds
    min_cv_score: float = 0.6
    min_consistency: float = 0.7

@dataclass
class CVResult:
    """Result of cross-validation."""
    method: CVMethod
    n_splits: int
    scores: Dict[str, List[float]]
    mean_scores: Dict[str, float]
    std_scores: Dict[str, float]
    best_model: str
    best_score: float
    validation_passed: bool
    processing_time: float
    detailed_results: Dict[str, Any]

class LabelCrossValidator:
    """Comprehensive cross-validation for triple barrier labels."""
    
    def __init__(self, config: Optional[CVConfig] = None):
        """Initialize the cross-validator.
        
        Args:
            config: Configuration for cross-validation
        """
        self.config = config or CVConfig()
        self.logger = logging.getLogger(f"{__name__}.LabelCrossValidator")
        
        # Initialize components
        self._initialize_components()
        
        # Performance tracking
        self.performance_stats = {
            'total_validations': 0,
            'total_processing_time': 0.0,
            'validation_success_rate': 0.0
        }
        
        self.logger.info("✅ LabelCrossValidator initialized successfully")

    def _initialize_components(self):
        """Initialize cross-validation components."""
        try:
            # Initialize common utilities
            self.common_utils = CommonUtilities()
            self.math_validator = MathValidation()
            
            # Initialize evaluation utilities
            self.evaluation_utils = EvaluationUtils()
            
            # Initialize CV components
            self.temporal_cv = TemporalCrossValidator(
                n_splits=self.config.n_splits,
                purged_pct=self.config.purged_pct
            )
            self.purged_cv = PurgedKFold(
                n_splits=self.config.n_splits,
                purged_pct=self.config.purged_pct
            )
            
            # Initialize models
            self.models = self._initialize_models()
            
            self.logger.info("✅ Cross-validation components initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize cross-validation components: {e}")
            raise

    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize machine learning models for validation."""
        models = {}
        
        if "random_forest" in self.config.models:
            models["random_forest"] = RandomForestClassifier(
                n_estimators=100,
                random_state=self.config.random_state,
                n_jobs=-1
            )
        
        if "logistic_regression" in self.config.models:
            models["logistic_regression"] = LogisticRegression(
                random_state=self.config.random_state,
                max_iter=1000
            )
        
        return models

    def validate_labels(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        labels_df: Optional[pd.DataFrame] = None,
        regime_data: Optional[pd.DataFrame] = None
    ) -> CVResult:
        """Validate labels using specified cross-validation method.
        
        Args:
            X: Feature matrix
            y: Label series
            labels_df: Full labels DataFrame (optional)
            regime_data: Regime information (optional)
            
        Returns:
            CVResult with validation scores and metrics
        """
        start_time = time.time()
        self.logger.info(f"🔄 Starting {self.config.method.value} validation")
        self.logger.info(f"📊 Features: {X.shape}, Labels: {y.shape}")
        
        try:
            # Validate inputs
            self._validate_inputs(X, y)
            
            # Select validation method
            if self.config.method == CVMethod.TEMPORAL_CV:
                scores, detailed_results = self._temporal_cv_validation(X, y)
            elif self.config.method == CVMethod.PURGED_CV:
                scores, detailed_results = self._purged_cv_validation(X, y)
            elif self.config.method == CVMethod.TIME_SERIES_CV:
                scores, detailed_results = self._time_series_cv_validation(X, y)
            elif self.config.method == CVMethod.REGIME_AWARE_CV:
                scores, detailed_results = self._regime_aware_cv_validation(X, y, regime_data)
            elif self.config.method == CVMethod.WALK_FORWARD_CV:
                scores, detailed_results = self._walk_forward_cv_validation(X, y)
            else:
                raise ValueError(f"Unsupported CV method: {self.config.method}")
            
            # Calculate summary statistics
            mean_scores = {metric: np.mean(scores[metric]) for metric in scores.keys()}
            std_scores = {metric: np.std(scores[metric]) for metric in scores.keys()}
            
            # Find best model
            best_model, best_score = self._find_best_model(mean_scores)
            
            # Determine if validation passed
            validation_passed = self._evaluate_validation_success(mean_scores)
            
            processing_time = time.time() - start_time
            
            result = CVResult(
                method=self.config.method,
                n_splits=self.config.n_splits,
                scores=scores,
                mean_scores=mean_scores,
                std_scores=std_scores,
                best_model=best_model,
                best_score=best_score,
                validation_passed=validation_passed,
                processing_time=processing_time,
                detailed_results=detailed_results
            )
            
            # Update performance stats
            self._update_performance_stats(processing_time, validation_passed)
            
            self.logger.info(f"✅ Validation completed in {processing_time:.3f}s")
            self.logger.info(f"🎯 Best model: {best_model} (score: {best_score:.3f})")
            self.logger.info(f"✅ Validation passed: {validation_passed}")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Validation failed after {processing_time:.3f}s: {e}")
            raise

    def _temporal_cv_validation(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
        """Perform temporal cross-validation."""
        self.logger.debug("🔄 Performing temporal cross-validation")
        
        scores = {metric.value: [] for metric in ValidationMetric}
        detailed_results = {'splits': []}
        
        try:
            # Get CV splits
            cv_splits = self.temporal_cv.split(X, y)
            
            for fold, (train_idx, test_idx) in enumerate(cv_splits):
                self.logger.debug(f"🔄 Processing fold {fold + 1}/{self.config.n_splits}")
                
                # Split data
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Train and evaluate models
                fold_scores = self._evaluate_fold(X_train, X_test, y_train, y_test)
                
                # Store scores
                for metric, score in fold_scores.items():
                    scores[metric].append(score)
                
                # Store detailed results
                detailed_results['splits'].append({
                    'fold': fold,
                    'train_size': len(train_idx),
                    'test_size': len(test_idx),
                    'scores': fold_scores
                })
            
            return scores, detailed_results
            
        except Exception as e:
            self.logger.error(f"❌ Temporal CV failed: {e}")
            raise

    def _purged_cv_validation(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
        """Perform purged cross-validation."""
        self.logger.debug("🔄 Performing purged cross-validation")
        
        scores = {metric.value: [] for metric in ValidationMetric}
        detailed_results = {'splits': []}
        
        try:
            # Get purged CV splits
            cv_splits = self.purged_cv.split(X, y)
            
            for fold, (train_idx, test_idx) in enumerate(cv_splits):
                self.logger.debug(f"🔄 Processing fold {fold + 1}/{self.config.n_splits}")
                
                # Split data
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Train and evaluate models
                fold_scores = self._evaluate_fold(X_train, X_test, y_train, y_test)
                
                # Store scores
                for metric, score in fold_scores.items():
                    scores[metric].append(score)
                
                # Store detailed results
                detailed_results['splits'].append({
                    'fold': fold,
                    'train_size': len(train_idx),
                    'test_size': len(test_idx),
                    'scores': fold_scores
                })
            
            return scores, detailed_results
            
        except Exception as e:
            self.logger.error(f"❌ Purged CV failed: {e}")
            raise

    def _time_series_cv_validation(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
        """Perform time series cross-validation."""
        self.logger.debug("🔄 Performing time series cross-validation")
        
        scores = {metric.value: [] for metric in ValidationMetric}
        detailed_results = {'splits': []}
        
        try:
            # Use sklearn's TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                self.logger.debug(f"🔄 Processing fold {fold + 1}/{self.config.n_splits}")
                
                # Split data
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Train and evaluate models
                fold_scores = self._evaluate_fold(X_train, X_test, y_train, y_test)
                
                # Store scores
                for metric, score in fold_scores.items():
                    scores[metric].append(score)
                
                # Store detailed results
                detailed_results['splits'].append({
                    'fold': fold,
                    'train_size': len(train_idx),
                    'test_size': len(test_idx),
                    'scores': fold_scores
                })
            
            return scores, detailed_results
            
        except Exception as e:
            self.logger.error(f"❌ Time series CV failed: {e}")
            raise

    def _regime_aware_cv_validation(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        regime_data: Optional[pd.DataFrame]
    ) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
        """Perform regime-aware cross-validation."""
        self.logger.debug("🔄 Performing regime-aware cross-validation")
        
        if regime_data is None:
            self.logger.warning("⚠️ No regime data provided, falling back to temporal CV")
            return self._temporal_cv_validation(X, y)
        
        scores = {metric.value: [] for metric in ValidationMetric}
        detailed_results = {'splits': [], 'regime_stats': {}}
        
        try:
            # Merge regime data
            merged_data = X.merge(regime_data, left_index=True, right_index=True, how='left')
            
            # Get unique regimes
            regimes = merged_data['regime'].unique()
            self.logger.info(f"📈 Found {len(regimes)} regimes for validation: {regimes}")
            
            # Validate each regime separately
            for regime in regimes:
                regime_mask = merged_data['regime'] == regime
                regime_X = X[regime_mask]
                regime_y = y[regime_mask]
                
                if len(regime_X) < self.config.min_samples_per_regime:
                    self.logger.warning(f"⚠️ Regime {regime} has insufficient samples: {len(regime_X)}")
                    continue
                
                self.logger.debug(f"🔄 Validating regime {regime} with {len(regime_X)} samples")
                
                # Perform temporal CV on regime data
                regime_scores, regime_results = self._temporal_cv_validation(regime_X, regime_y)
                
                # Aggregate scores
                for metric, score_list in regime_scores.items():
                    scores[metric].extend(score_list)
                
                # Store regime statistics
                detailed_results['regime_stats'][regime] = {
                    'sample_count': len(regime_X),
                    'mean_scores': {metric: np.mean(score_list) for metric, score_list in regime_scores.items()},
                    'splits': regime_results['splits']
                }
            
            return scores, detailed_results
            
        except Exception as e:
            self.logger.error(f"❌ Regime-aware CV failed: {e}")
            raise

    def _walk_forward_cv_validation(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
        """Perform walk-forward cross-validation."""
        self.logger.debug("🔄 Performing walk-forward cross-validation")
        
        scores = {metric.value: [] for metric in ValidationMetric}
        detailed_results = {'splits': []}
        
        try:
            total_samples = len(X)
            initial_size = self.config.initial_train_size
            step_size = self.config.step_size
            
            if total_samples < initial_size + step_size:
                raise ValueError(f"Insufficient data for walk-forward CV: {total_samples} < {initial_size + step_size}")
            
            fold = 0
            start_idx = initial_size
            
            while start_idx + step_size <= total_samples:
                # Define train and test sets
                train_end = start_idx
                test_start = start_idx
                test_end = min(start_idx + step_size, total_samples)
                
                train_idx = range(0, train_end)
                test_idx = range(test_start, test_end)
                
                self.logger.debug(f"🔄 Processing fold {fold + 1}: train={len(train_idx)}, test={len(test_idx)}")
                
                # Split data
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Train and evaluate models
                fold_scores = self._evaluate_fold(X_train, X_test, y_train, y_test)
                
                # Store scores
                for metric, score in fold_scores.items():
                    scores[metric].append(score)
                
                # Store detailed results
                detailed_results['splits'].append({
                    'fold': fold,
                    'train_size': len(train_idx),
                    'test_size': len(test_idx),
                    'train_range': (0, train_end),
                    'test_range': (test_start, test_end),
                    'scores': fold_scores
                })
                
                fold += 1
                start_idx += step_size
            
            return scores, detailed_results
            
        except Exception as e:
            self.logger.error(f"❌ Walk-forward CV failed: {e}")
            raise

    def _evaluate_fold(
        self, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame, 
        y_train: pd.Series, 
        y_test: pd.Series
    ) -> Dict[str, float]:
        """Evaluate a single fold with all models."""
        fold_scores = {}
        
        try:
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Evaluate each model
            for model_name, model in self.models.items():
                try:
                    # Train model
                    if model_name == "logistic_regression":
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    # Store scores (use model name as prefix)
                    fold_scores[f"{model_name}_accuracy"] = accuracy
                    fold_scores[f"{model_name}_precision"] = precision
                    fold_scores[f"{model_name}_recall"] = recall
                    fold_scores[f"{model_name}_f1_score"] = f1
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Model {model_name} failed in fold: {e}")
                    fold_scores[f"{model_name}_accuracy"] = 0.0
                    fold_scores[f"{model_name}_precision"] = 0.0
                    fold_scores[f"{model_name}_recall"] = 0.0
                    fold_scores[f"{model_name}_f1_score"] = 0.0
            
            # Calculate average scores across models
            fold_scores['accuracy'] = np.mean([v for k, v in fold_scores.items() if k.endswith('_accuracy')])
            fold_scores['precision'] = np.mean([v for k, v in fold_scores.items() if k.endswith('_precision')])
            fold_scores['recall'] = np.mean([v for k, v in fold_scores.items() if k.endswith('_recall')])
            fold_scores['f1_score'] = np.mean([v for k, v in fold_scores.items() if k.endswith('_f1_score')])
            
            return fold_scores
            
        except Exception as e:
            self.logger.error(f"❌ Fold evaluation failed: {e}")
            return {metric.value: 0.0 for metric in ValidationMetric}

    def _find_best_model(self, mean_scores: Dict[str, float]) -> Tuple[str, float]:
        """Find the best performing model."""
        try:
            # Look for model-specific scores
            model_scores = {}
            for metric, score in mean_scores.items():
                if '_' in metric and not metric.endswith('_score'):
                    model_name = metric.split('_')[0]
                    if model_name not in model_scores:
                        model_scores[model_name] = []
                    model_scores[model_name].append(score)
            
            if not model_scores:
                return "unknown", 0.0
            
            # Calculate average score for each model
            model_avg_scores = {
                model: np.mean(scores) for model, scores in model_scores.items()
            }
            
            # Find best model
            best_model = max(model_avg_scores, key=model_avg_scores.get)
            best_score = model_avg_scores[best_model]
            
            return best_model, best_score
            
        except Exception as e:
            self.logger.error(f"❌ Failed to find best model: {e}")
            return "unknown", 0.0

    def _evaluate_validation_success(self, mean_scores: Dict[str, float]) -> bool:
        """Evaluate if validation was successful."""
        try:
            # Check if accuracy meets minimum threshold
            accuracy = mean_scores.get('accuracy', 0.0)
            if accuracy < self.config.min_cv_score:
                return False
            
            # Check consistency across metrics
            scores = [v for k, v in mean_scores.items() if k in ['accuracy', 'precision', 'recall', 'f1_score']]
            if not scores:
                return False
            
            consistency = 1.0 - (np.std(scores) / (np.mean(scores) + 1e-10))
            if consistency < self.config.min_consistency:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to evaluate validation success: {e}")
            return False

    def _validate_inputs(self, X: pd.DataFrame, y: pd.Series):
        """Validate input data for cross-validation."""
        if len(X) != len(y):
            raise ValueError(f"Feature and label lengths don't match: {len(X)} vs {len(y)}")
        
        if len(X) < 50:
            raise ValueError(f"Insufficient data for cross-validation: {len(X)} < 50")
        
        if X.isnull().any().any():
            raise ValueError("Features contain null values")
        
        if y.isnull().any():
            raise ValueError("Labels contain null values")

    def _update_performance_stats(self, processing_time: float, validation_passed: bool):
        """Update performance statistics."""
        self.performance_stats['total_validations'] += 1
        self.performance_stats['total_processing_time'] += processing_time
        
        if validation_passed:
            self.performance_stats['validation_success_rate'] = (
                self.performance_stats['validation_success_rate'] * (self.performance_stats['total_validations'] - 1) + 1.0
            ) / self.performance_stats['total_validations']
        else:
            self.performance_stats['validation_success_rate'] = (
                self.performance_stats['validation_success_rate'] * (self.performance_stats['total_validations'] - 1) + 0.0
            ) / self.performance_stats['total_validations']

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            'total_validations': self.performance_stats['total_validations'],
            'total_processing_time': self.performance_stats['total_processing_time'],
            'validation_success_rate': self.performance_stats['validation_success_rate'],
            'average_processing_time': safe_divide(
                self.performance_stats['total_processing_time'],
                self.performance_stats['total_validations']
            )
        }

    def generate_validation_report(self, result: CVResult) -> str:
        """Generate a detailed validation report."""
        report = []
        report.append("=" * 60)
        report.append("TRIPLE BARRIER LABEL CROSS-VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Method: {result.method.value}")
        report.append(f"Number of Splits: {result.n_splits}")
        report.append(f"Processing Time: {result.processing_time:.3f} seconds")
        report.append("")
        
        # Overall results
        report.append("OVERALL RESULTS:")
        report.append("-" * 30)
        report.append(f"Best Model: {result.best_model}")
        report.append(f"Best Score: {result.best_score:.3f}")
        report.append(f"Validation Passed: {'✅ Yes' if result.validation_passed else '❌ No'}")
        report.append("")
        
        # Mean scores
        report.append("MEAN SCORES:")
        report.append("-" * 30)
        for metric, score in result.mean_scores.items():
            std_score = result.std_scores.get(metric, 0.0)
            report.append(f"{metric.replace('_', ' ').title()}: {score:.3f} ± {std_score:.3f}")
        report.append("")
        
        # Detailed results
        if 'splits' in result.detailed_results:
            report.append("FOLD DETAILS:")
            report.append("-" * 30)
            for split in result.detailed_results['splits']:
                report.append(f"Fold {split['fold'] + 1}:")
                report.append(f"  Train Size: {split['train_size']}")
                report.append(f"  Test Size: {split['test_size']}")
                if 'scores' in split:
                    for metric, score in split['scores'].items():
                        if not metric.endswith('_') and not metric.startswith('_'):
                            report.append(f"  {metric}: {score:.3f}")
                report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)