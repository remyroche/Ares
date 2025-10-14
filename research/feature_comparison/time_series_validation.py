"""
Time-Series Safe Validation and Cross-Validation

This module provides time-series safe validation methods including purged,
embargoed CV, walk-forward analysis, and out-of-sample testing.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

logger = logging.getLogger(__name__)

class PurgedGroupKFold(BaseCrossValidator):
    """
    Purged Group K-Fold with embargo for time-series data.
    
    This prevents data leakage by:
    1. Purging samples where target overlaps with training data
    2. Adding embargo period between train/test splits
    """
    
    def __init__(self, n_splits: int = 5, embargo_periods: int = 1):
        """
        Initialize PurgedGroupKFold.
        
        Args:
            n_splits: Number of splits
            embargo_periods: Number of periods to embargo between train/test
        """
        self.n_splits = n_splits
        self.embargo_periods = embargo_periods
    
    def split(self, X, y=None, groups=None):
        """
        Generate train/test splits with purging and embargo.
        
        Args:
            X: Feature matrix
            y: Target vector
            groups: Group labels (e.g., timestamps)
            
        Yields:
            (train_idx, test_idx) tuples
        """
        if groups is None:
            groups = np.arange(len(X))
        
        # Sort by groups (time)
        sorted_indices = np.argsort(groups)
        groups_sorted = groups[sorted_indices]
        
        # Create time-based splits
        n_samples = len(X)
        split_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            # Test set boundaries
            test_start = i * split_size
            test_end = min((i + 1) * split_size, n_samples)
            
            # Purge training set (remove overlapping targets)
            train_indices = []
            test_indices = list(range(test_start, test_end))
            
            # Add embargo periods
            embargo_start = max(0, test_start - self.embargo_periods)
            embargo_end = min(n_samples, test_end + self.embargo_periods)
            
            # Training indices (before embargo start and after embargo end)
            train_indices.extend(range(0, embargo_start))
            train_indices.extend(range(embargo_end, n_samples))
            
            # Convert back to original indices
            train_idx = sorted_indices[train_indices]
            test_idx = sorted_indices[test_indices]
            
            yield train_idx, test_idx
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Return number of splits."""
        return self.n_splits

class WalkForwardValidator:
    """
    Walk-forward validation for time-series data.
    
    Mirrors deployment latency by training on past data and testing on future data.
    """
    
    def __init__(self, train_window: int = 1000, test_window: int = 100, 
                 step_size: int = 50, min_train_samples: int = 500):
        """
        Initialize walk-forward validator.
        
        Args:
            train_window: Size of training window
            test_window: Size of test window
            step_size: Step size for moving window
            min_train_samples: Minimum samples required for training
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.min_train_samples = min_train_samples
    
    def split(self, X, y=None, groups=None):
        """
        Generate walk-forward splits.
        
        Args:
            X: Feature matrix
            y: Target vector
            groups: Group labels (e.g., timestamps)
            
        Yields:
            (train_idx, test_idx) tuples
        """
        n_samples = len(X)
        
        for start in range(0, n_samples - self.train_window - self.test_window + 1, self.step_size):
            train_end = start + self.train_window
            test_start = train_end
            test_end = min(test_start + self.test_window, n_samples)
            
            # Ensure we have enough training samples
            if train_end - start < self.min_train_samples:
                continue
            
            train_idx = list(range(start, train_end))
            test_idx = list(range(test_start, test_end))
            
            yield train_idx, test_idx
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Return number of splits."""
        n_samples = len(X) if X is not None else 1000
        return max(1, (n_samples - self.train_window - self.test_window) // self.step_size + 1)

class OutOfSampleValidator:
    """
    Out-of-sample validation for different assets/regimes.
    """
    
    def __init__(self, asset_column: str = 'asset', regime_column: str = 'regime'):
        """
        Initialize out-of-sample validator.
        
        Args:
            asset_column: Column name for asset identification
            regime_column: Column name for regime identification
        """
        self.asset_column = asset_column
        self.regime_column = regime_column
    
    def split_by_asset(self, X, y=None, groups=None):
        """
        Split by asset: train on one asset, test on another.
        
        Args:
            X: Feature matrix
            y: Target vector
            groups: DataFrame with asset information
            
        Yields:
            (train_idx, test_idx) tuples
        """
        if groups is None or self.asset_column not in groups.columns:
            logger.warning("No asset information available for out-of-asset validation")
            return
        
        assets = groups[self.asset_column].unique()
        
        for train_asset in assets:
            for test_asset in assets:
                if train_asset == test_asset:
                    continue
                
                train_mask = groups[self.asset_column] == train_asset
                test_mask = groups[self.asset_column] == test_asset
                
                train_idx = np.where(train_mask)[0]
                test_idx = np.where(test_mask)[0]
                
                if len(train_idx) > 0 and len(test_idx) > 0:
                    yield train_idx, test_idx
    
    def split_by_regime(self, X, y=None, groups=None):
        """
        Split by regime: train on one regime, test on another.
        
        Args:
            X: Feature matrix
            y: Target vector
            groups: DataFrame with regime information
            
        Yields:
            (train_idx, test_idx) tuples
        """
        if groups is None or self.regime_column not in groups.columns:
            logger.warning("No regime information available for out-of-regime validation")
            return
        
        regimes = groups[self.regime_column].unique()
        
        for train_regime in regimes:
            for test_regime in regimes:
                if train_regime == test_regime:
                    continue
                
                train_mask = groups[self.regime_column] == train_regime
                test_mask = groups[self.regime_column] == test_regime
                
                train_idx = np.where(train_mask)[0]
                test_idx = np.where(test_mask)[0]
                
                if len(train_idx) > 0 and len(test_idx) > 0:
                    yield train_idx, test_idx

class TimeSeriesValidator:
    """
    Comprehensive time-series validation orchestrator.
    """
    
    def __init__(self, n_splits: int = 5, embargo_periods: int = 1,
                 train_window: int = 1000, test_window: int = 100,
                 step_size: int = 50):
        """
        Initialize time-series validator.
        
        Args:
            n_splits: Number of CV splits
            embargo_periods: Embargo periods between train/test
            train_window: Training window size
            test_window: Test window size
            step_size: Step size for walk-forward
        """
        self.n_splits = n_splits
        self.embargo_periods = embargo_periods
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        
        # Initialize validators
        self.purged_cv = PurgedGroupKFold(n_splits, embargo_periods)
        self.walk_forward = WalkForwardValidator(train_window, test_window, step_size)
        self.out_of_sample = OutOfSampleValidator()
    
    def validate_model(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                      groups: Optional[pd.DataFrame] = None,
                      validation_type: str = 'purged_cv') -> Dict[str, Any]:
        """
        Validate model using specified validation method.
        
        Args:
            model: Model to validate
            X: Feature matrix
            y: Target vector
            groups: Group information (timestamps, assets, regimes)
            validation_type: Type of validation ('purged_cv', 'walk_forward', 'out_of_asset', 'out_of_regime')
            
        Returns:
            Validation results
        """
        logger.info(f"Running {validation_type} validation...")
        
        # Select validator
        if validation_type == 'purged_cv':
            validator = self.purged_cv
            group_col = groups['timestamp'] if groups is not None else None
        elif validation_type == 'walk_forward':
            validator = self.walk_forward
            group_col = None
        elif validation_type == 'out_of_asset':
            validator = self.out_of_sample
            group_col = groups
        elif validation_type == 'out_of_regime':
            validator = self.out_of_sample
            group_col = groups
        else:
            raise ValueError(f"Unknown validation type: {validation_type}")
        
        # Run validation
        scores = []
        feature_importances = []
        
        for train_idx, test_idx in validator.split(X, y, group_col):
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate scores
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            scores.append({
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'n_train': len(train_idx),
                'n_test': len(test_idx)
            })
            
            # Store feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_importances.append(model.feature_importances_)
        
        # Aggregate results
        results = {
            'validation_type': validation_type,
            'n_splits': len(scores),
            'scores': scores,
            'mean_scores': {
                'mse': np.mean([s['mse'] for s in scores]),
                'mae': np.mean([s['mae'] for s in scores]),
                'r2': np.mean([s['r2'] for s in scores])
            },
            'std_scores': {
                'mse': np.std([s['mse'] for s in scores]),
                'mae': np.std([s['mae'] for s in scores]),
                'r2': np.std([s['r2'] for s in scores])
            }
        }
        
        # Add feature importance if available
        if feature_importances:
            feature_importance_df = pd.DataFrame(feature_importances, columns=X.columns)
            results['feature_importances'] = {
                'mean': feature_importance_df.mean(),
                'std': feature_importance_df.std(),
                'cv': feature_importance_df.std() / (feature_importance_df.mean() + 1e-8)
            }
        
        return results
    
    def run_all_validations(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                           groups: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Run all validation methods.
        
        Args:
            model: Model to validate
            X: Feature matrix
            y: Target vector
            groups: Group information
            
        Returns:
            All validation results
        """
        all_results = {}
        
        # Purged CV
        try:
            all_results['purged_cv'] = self.validate_model(
                model, X, y, groups, 'purged_cv'
            )
        except Exception as e:
            logger.warning(f"Purged CV failed: {e}")
            all_results['purged_cv'] = {'error': str(e)}
        
        # Walk-forward
        try:
            all_results['walk_forward'] = self.validate_model(
                model, X, y, groups, 'walk_forward'
            )
        except Exception as e:
            logger.warning(f"Walk-forward validation failed: {e}")
            all_results['walk_forward'] = {'error': str(e)}
        
        # Out-of-asset (if asset info available)
        if groups is not None and 'asset' in groups.columns:
            try:
                all_results['out_of_asset'] = self.validate_model(
                    model, X, y, groups, 'out_of_asset'
                )
            except Exception as e:
                logger.warning(f"Out-of-asset validation failed: {e}")
                all_results['out_of_asset'] = {'error': str(e)}
        
        # Out-of-regime (if regime info available)
        if groups is not None and 'regime' in groups.columns:
            try:
                all_results['out_of_regime'] = self.validate_model(
                    model, X, y, groups, 'out_of_regime'
                )
            except Exception as e:
                logger.warning(f"Out-of-regime validation failed: {e}")
                all_results['out_of_regime'] = {'error': str(e)}
        
        return all_results