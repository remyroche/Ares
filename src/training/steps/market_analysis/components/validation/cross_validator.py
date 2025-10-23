"""
Cross Validator for Market Analysis Components.

This module provides cross-validation capabilities for market analysis
pipeline steps, including time series cross-validation, walk-forward
validation, and regime-aware validation.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error
from src.utils.common_utilities import safe_dataframe_operation, validate_dataframe_columns
from src.utils.math_validation import validate_finite, safe_divide
from src.training.steps.market_analysis.components.base_component import BaseMarketAnalysisComponent, ComponentConfig

class CrossValidationType(Enum):
    """Types of cross-validation."""
    TIME_SERIES = "time_series"
    WALK_FORWARD = "walk_forward"
    REGIME_AWARE = "regime_aware"
    BLOCKING = "blocking"

@dataclass
class CrossValidationConfig:
    """Configuration for cross-validation."""
    # General settings
    cv_type: CrossValidationType = CrossValidationType.TIME_SERIES
    n_splits: int = 5
    test_size: float = 0.2
    gap_size: int = 0
    
    # Time series specific
    min_train_size: int = 100
    max_train_size: Optional[int] = None
    
    # Walk-forward specific
    step_size: int = 1
    expanding_window: bool = True
    
    # Regime-aware specific
    regime_column: str = "regime"
    min_regime_samples: int = 20
    
    # Validation metrics
    primary_metric: str = "accuracy"
    secondary_metrics: List[str] = field(default_factory=lambda: ["precision", "recall", "f1"])
    
    # Performance settings
    parallel: bool = False
    n_jobs: int = -1

@dataclass
class CrossValidationResult:
    """Result of cross-validation."""
    cv_scores: Dict[str, List[float]]
    mean_scores: Dict[str, float]
    std_scores: Dict[str, float]
    best_params: Dict[str, Any]
    validation_details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

class CrossValidator(BaseMarketAnalysisComponent):
    """
    Cross-validator for market analysis components.
    
    Provides various cross-validation strategies:
    - Time series cross-validation
    - Walk-forward validation
    - Regime-aware validation
    - Blocking validation
    """
    
    def __init__(self, config: Optional[CrossValidationConfig] = None):
        """Initialize the cross validator."""
        super().__init__(ComponentConfig())
        self.cv_config = config or CrossValidationConfig()
        self.logger = logging.getLogger(__name__)
        
    async def cross_validate(self, 
                           data: pd.DataFrame,
                           model_func: Callable,
                           target_column: str,
                           feature_columns: List[str],
                           context: str = "cross_validation") -> CrossValidationResult:
        """
        Perform cross-validation on market analysis model.
        
        Args:
            data: Market data DataFrame
            model_func: Function that trains and evaluates model
            target_column: Name of target column
            feature_columns: List of feature column names
            context: Validation context for logging
            
        Returns:
            CrossValidationResult with validation scores and details
        """
        try:
            tprint_info(f"🔍 Starting {self.cv_config.cv_type.value} cross-validation for {context}")
            
            # Initialize result
            result = CrossValidationResult(
                cv_scores={},
                mean_scores={},
                std_scores={},
                best_params={}
            )
            
            # Prepare data
            X, y = self._prepare_data(data, feature_columns, target_column)
            
            # Perform cross-validation based on type
            if self.cv_config.cv_type == CrossValidationType.TIME_SERIES:
                await self._time_series_cv(X, y, model_func, result)
            elif self.cv_config.cv_type == CrossValidationType.WALK_FORWARD:
                await self._walk_forward_cv(data, X, y, model_func, result)
            elif self.cv_config.cv_type == CrossValidationType.REGIME_AWARE:
                await self._regime_aware_cv(data, X, y, model_func, result)
            elif self.cv_config.cv_type == CrossValidationType.BLOCKING:
                await self._blocking_cv(X, y, model_func, result)
            
            # Calculate summary statistics
            self._calculate_summary_stats(result)
            
            tprint_info(f"✅ Cross-validation completed: {len(result.cv_scores)} metrics evaluated")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Cross-validation failed: {str(e)}")
            return CrossValidationResult(
                cv_scores={},
                mean_scores={},
                std_scores={},
                best_params={},
                errors=[str(e)]
            )
    
    def _prepare_data(self, data: pd.DataFrame, feature_columns: List[str], target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for cross-validation."""
        try:
            # Select features and target
            X = data[feature_columns].values
            y = data[target_column].values
            
            # Remove NaN values
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_mask]
            y = y[valid_mask]
            
            return X, y
            
        except Exception as e:
            raise ValueError(f"Data preparation failed: {str(e)}")
    
    async def _time_series_cv(self, X: np.ndarray, y: np.ndarray, model_func: Callable, result: CrossValidationResult):
        """Perform time series cross-validation."""
        try:
            tscv = TimeSeriesSplit(
                n_splits=self.cv_config.n_splits,
                test_size=int(len(X) * self.cv_config.test_size),
                gap=self.cv_config.gap_size
            )
            
            cv_scores = {metric: [] for metric in [self.cv_config.primary_metric] + self.cv_config.secondary_metrics}
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                try:
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    # Train and evaluate model
                    fold_scores = await model_func(X_train, X_test, y_train, y_test)
                    
                    # Store scores
                    for metric, score in fold_scores.items():
                        if metric in cv_scores:
                            cv_scores[metric].append(score)
                    
                    tprint_info(f"Fold {fold + 1}/{self.cv_config.n_splits} completed")
                    
                except Exception as e:
                    result.warnings.append(f"Fold {fold + 1} failed: {str(e)}")
                    continue
            
            result.cv_scores = cv_scores
            
        except Exception as e:
            result.errors.append(f"Time series CV failed: {str(e)}")
    
    async def _walk_forward_cv(self, data: pd.DataFrame, X: np.ndarray, y: np.ndarray, model_func: Callable, result: CrossValidationResult):
        """Perform walk-forward cross-validation."""
        try:
            n_samples = len(X)
            min_train_size = self.cv_config.min_train_size
            step_size = self.cv_config.step_size
            
            cv_scores = {metric: [] for metric in [self.cv_config.primary_metric] + self.cv_config.secondary_metrics}
            
            # Calculate test size
            test_size = int(n_samples * self.cv_config.test_size)
            
            fold = 0
            for start_idx in range(min_train_size, n_samples - test_size, step_size):
                try:
                    if self.cv_config.expanding_window:
                        # Expanding window
                        train_end = start_idx
                        train_start = 0
                    else:
                        # Rolling window
                        train_start = start_idx - min_train_size
                        train_end = start_idx
                    
                    test_start = start_idx
                    test_end = min(start_idx + test_size, n_samples)
                    
                    X_train = X[train_start:train_end]
                    X_test = X[test_start:test_end]
                    y_train = y[train_start:train_end]
                    y_test = y[test_start:test_end]
                    
                    # Train and evaluate model
                    fold_scores = await model_func(X_train, X_test, y_train, y_test)
                    
                    # Store scores
                    for metric, score in fold_scores.items():
                        if metric in cv_scores:
                            cv_scores[metric].append(score)
                    
                    fold += 1
                    tprint_info(f"Walk-forward fold {fold} completed")
                    
                except Exception as e:
                    result.warnings.append(f"Walk-forward fold {fold} failed: {str(e)}")
                    continue
            
            result.cv_scores = cv_scores
            
        except Exception as e:
            result.errors.append(f"Walk-forward CV failed: {str(e)}")
    
    async def _regime_aware_cv(self, data: pd.DataFrame, X: np.ndarray, y: np.ndarray, model_func: Callable, result: CrossValidationResult):
        """Perform regime-aware cross-validation."""
        try:
            if self.cv_config.regime_column not in data.columns:
                result.errors.append(f"Regime column '{self.cv_config.regime_column}' not found")
                return
            
            regime_assignments = data[self.cv_config.regime_column].values
            unique_regimes = np.unique(regime_assignments)
            
            cv_scores = {metric: [] for metric in [self.cv_config.primary_metric] + self.cv_config.secondary_metrics}
            
            # Create regime-based splits
            for regime in unique_regimes:
                regime_mask = regime_assignments == regime
                regime_indices = np.where(regime_mask)[0]
                
                if len(regime_indices) < self.cv_config.min_regime_samples:
                    result.warnings.append(f"Regime {regime} has insufficient samples: {len(regime_indices)}")
                    continue
                
                # Split regime data
                regime_X = X[regime_indices]
                regime_y = y[regime_indices]
                
                # Use time series split for regime data
                tscv = TimeSeriesSplit(n_splits=min(3, len(regime_indices) // 20))
                
                for fold, (train_idx, test_idx) in enumerate(tscv.split(regime_X)):
                    try:
                        X_train = regime_X[train_idx]
                        X_test = regime_X[test_idx]
                        y_train = regime_y[train_idx]
                        y_test = regime_y[test_idx]
                        
                        # Train and evaluate model
                        fold_scores = await model_func(X_train, X_test, y_train, y_test)
                        
                        # Store scores
                        for metric, score in fold_scores.items():
                            if metric in cv_scores:
                                cv_scores[metric].append(score)
                        
                    except Exception as e:
                        result.warnings.append(f"Regime {regime} fold {fold} failed: {str(e)}")
                        continue
            
            result.cv_scores = cv_scores
            result.validation_details['regime_breakdown'] = {
                'unique_regimes': len(unique_regimes),
                'regime_samples': {regime: np.sum(regime_assignments == regime) for regime in unique_regimes}
            }
            
        except Exception as e:
            result.errors.append(f"Regime-aware CV failed: {str(e)}")
    
    async def _blocking_cv(self, X: np.ndarray, y: np.ndarray, model_func: Callable, result: CrossValidationResult):
        """Perform blocking cross-validation."""
        try:
            n_samples = len(X)
            block_size = n_samples // self.cv_config.n_splits
            
            cv_scores = {metric: [] for metric in [self.cv_config.primary_metric] + self.cv_config.secondary_metrics}
            
            for fold in range(self.cv_config.n_splits):
                try:
                    # Create block-based split
                    test_start = fold * block_size
                    test_end = min((fold + 1) * block_size, n_samples)
                    
                    # Create training set (all data except test block)
                    train_indices = list(range(0, test_start)) + list(range(test_end, n_samples))
                    
                    if len(train_indices) == 0:
                        result.warnings.append(f"Fold {fold} has no training data")
                        continue
                    
                    X_train = X[train_indices]
                    X_test = X[test_start:test_end]
                    y_train = y[train_indices]
                    y_test = y[test_start:test_end]
                    
                    # Train and evaluate model
                    fold_scores = await model_func(X_train, X_test, y_train, y_test)
                    
                    # Store scores
                    for metric, score in fold_scores.items():
                        if metric in cv_scores:
                            cv_scores[metric].append(score)
                    
                    tprint_info(f"Blocking fold {fold + 1}/{self.cv_config.n_splits} completed")
                    
                except Exception as e:
                    result.warnings.append(f"Blocking fold {fold} failed: {str(e)}")
                    continue
            
            result.cv_scores = cv_scores
            
        except Exception as e:
            result.errors.append(f"Blocking CV failed: {str(e)}")
    
    def _calculate_summary_stats(self, result: CrossValidationResult):
        """Calculate summary statistics for cross-validation results."""
        try:
            for metric, scores in result.cv_scores.items():
                if scores:
                    result.mean_scores[metric] = np.mean(scores)
                    result.std_scores[metric] = np.std(scores)
                else:
                    result.mean_scores[metric] = 0.0
                    result.std_scores[metric] = 0.0
            
            # Find best parameters (simplified - would need more sophisticated parameter search)
            if result.mean_scores:
                best_metric = max(result.mean_scores.keys(), key=lambda k: result.mean_scores[k])
                result.best_params = {
                    'best_metric': best_metric,
                    'best_score': result.mean_scores[best_metric],
                    'cv_type': self.cv_config.cv_type.value
                }
            
        except Exception as e:
            result.errors.append(f"Summary statistics calculation failed: {str(e)}")
    
    def get_validation_summary(self, result: CrossValidationResult) -> Dict[str, Any]:
        """Get a summary of cross-validation results."""
        return {
            'cv_type': self.cv_config.cv_type.value,
            'n_splits': self.cv_config.n_splits,
            'mean_scores': result.mean_scores,
            'std_scores': result.std_scores,
            'best_params': result.best_params,
            'n_warnings': len(result.warnings),
            'n_errors': len(result.errors),
            'validation_quality': 'good' if len(result.errors) == 0 else 'poor'
        }