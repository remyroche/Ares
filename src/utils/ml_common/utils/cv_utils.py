"""
Cross-Validation Utilities

This module provides utilities for cross-validation with memory-aware operations.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

class CrossValidationUtilities:
    """Cross-validation utilities with memory management."""

    def __init__(self):
        """Initialize CV utilities."""
        self.logger = logger.getChild('CrossValidationUtilities')
        self.logger.info("🚀 Initializing CrossValidationUtilities")

    def walk_forward_validation(
        self,
        data: pd.DataFrame,
        model_function: Callable,
        target_column: str,
        n_splits: int = 5,
        test_size: int = 30,
        gap: int = 1,
        metric_functions: Optional[List[Callable]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform walk-forward validation for time series data.

        Args:
            data: Time series data
            model_function: Function that returns a fitted model
            target_column: Name of target column
            n_splits: Number of validation splits
            test_size: Size of test set for each split
            gap: Gap between train and test sets
            metric_functions: List of metric functions to evaluate

        Returns:
            Dictionary containing validation results
        """
        self.logger.info(f"🔍 Starting walk-forward validation with {n_splits} splits")

        start_time = time.time()

        if metric_functions is None:
            metric_functions = [
                lambda y_true, y_pred: mean_squared_error(y_true, y_pred),
                lambda y_true, y_pred: mean_absolute_error(y_true, y_pred),
                lambda y_true, y_pred: r2_score(y_true, y_pred)
            ]

        # Initialize results storage
        fold_results = []
        predictions = []
        actuals = []

        # Create time series split
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)

        data_length = len(data)
        self.logger.debug(f"📊 Data length: {data_length}, n_splits: {n_splits}")

        for fold, (train_index, test_index) in enumerate(tscv.split(data)):
            self.logger.debug(f"📈 Processing fold {fold + 1}/{n_splits}")

            # Split data
            train_data = data.iloc[train_index]
            test_data = data.iloc[test_index]

            if len(train_data) == 0 or len(test_data) == 0:
                self.logger.warning(f"⚠️ Empty train or test set in fold {fold + 1}, skipping")
                continue

            # Prepare features and target
            X_train = train_data.drop(columns=[target_column])
            y_train = train_data[target_column]
            X_test = test_data.drop(columns=[target_column])
            y_test = test_data[target_column]

            try:
                # Train model
                model = model_function(X_train, y_train, **kwargs)

                # Make predictions
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_test)
                else:
                    # Assume model is a function
                    y_pred = model(X_test)

                # Calculate metrics
                fold_metrics = {}
                for i, metric_func in enumerate(metric_functions):
                    try:
                        metric_name = f'metric_{i}'
                        metric_value = metric_func(y_test, y_pred)
                        fold_metrics[metric_name] = metric_value
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to calculate metric {i}: {e}")
                        fold_metrics[f'metric_{i}'] = float('nan')

                # Store fold results
                fold_result = {
                    'fold': fold + 1,
                    'train_size': len(train_data),
                    'test_size': len(test_data),
                    'metrics': fold_metrics,
                    'predictions': y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred),
                    'actuals': y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test)
                }
                fold_results.append(fold_result)

                # Store predictions and actuals for overall metrics
                predictions.extend(y_pred)
                actuals.extend(y_test)

            except Exception as e:
                self.logger.error(f"❌ Fold {fold + 1} failed: {e}")
                fold_result = {
                    'fold': fold + 1,
                    'train_size': len(train_data),
                    'test_size': len(test_data),
                    'error': str(e),
                    'success': False
                }
                fold_results.append(fold_result)

        # Calculate overall metrics
        overall_metrics = {}
        if predictions and actuals:
            for i, metric_func in enumerate(metric_functions):
                try:
                    overall_metrics[f'overall_metric_{i}'] = metric_func(actuals, predictions)
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to calculate overall metric {i}: {e}")
                    overall_metrics[f'overall_metric_{i}'] = float('nan')

        validation_time = time.time() - start_time

        result = {
            'fold_results': fold_results,
            'overall_metrics': overall_metrics,
            'n_splits': n_splits,
            'test_size': test_size,
            'gap': gap,
            'total_samples': len(data),
            'validation_time': validation_time,
            'success': len(fold_results) > 0 and any(r.get('success', True) for r in fold_results)
        }

        self.logger.info(f"✅ Walk-forward validation completed in {validation_time:.2f}s")
        return result

    def k_fold_cross_validation(
        self,
        data: pd.DataFrame,
        model_function: Callable,
        target_column: str,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        metric_functions: Optional[List[Callable]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation.

        Args:
            data: Dataset for cross-validation
            model_function: Function that returns a fitted model
            target_column: Name of target column
            n_splits: Number of CV folds
            shuffle: Whether to shuffle data before splitting
            random_state: Random state for reproducibility
            metric_functions: List of metric functions to evaluate

        Returns:
            Dictionary containing validation results
        """
        self.logger.info(f"🔍 Starting {n_splits}-fold cross-validation")

        start_time = time.time()

        if metric_functions is None:
            metric_functions = [
                lambda y_true, y_pred: mean_squared_error(y_true, y_pred),
                lambda y_true, y_pred: mean_absolute_error(y_true, y_pred),
                lambda y_true, y_pred: r2_score(y_true, y_pred)
            ]

        from sklearn.model_selection import KFold

        # Initialize results storage
        fold_results = []
        predictions = []
        actuals = []

        # Create k-fold split
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        X = data.drop(columns=[target_column])
        y = data[target_column]

        for fold, (train_index, test_index) in enumerate(kf.split(X)):
            self.logger.debug(f"📈 Processing fold {fold + 1}/{n_splits}")

            # Split data
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            try:
                # Train model
                model = model_function(X_train, y_train, **kwargs)

                # Make predictions
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_test)
                else:
                    # Assume model is a function
                    y_pred = model(X_test)

                # Calculate metrics
                fold_metrics = {}
                for i, metric_func in enumerate(metric_functions):
                    try:
                        metric_value = metric_func(y_test, y_pred)
                        fold_metrics[f'metric_{i}'] = metric_value
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to calculate metric {i}: {e}")
                        fold_metrics[f'metric_{i}'] = float('nan')

                # Store fold results
                fold_result = {
                    'fold': fold + 1,
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'metrics': fold_metrics,
                    'predictions': y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred),
                    'actuals': y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test)
                }
                fold_results.append(fold_result)

                # Store predictions and actuals for overall metrics
                predictions.extend(y_pred)
                actuals.extend(y_test)

            except Exception as e:
                self.logger.error(f"❌ Fold {fold + 1} failed: {e}")
                fold_result = {
                    'fold': fold + 1,
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'error': str(e),
                    'success': False
                }
                fold_results.append(fold_result)

        # Calculate overall metrics
        overall_metrics = {}
        if predictions and actuals:
            for i, metric_func in enumerate(metric_functions):
                try:
                    overall_metrics[f'overall_metric_{i}'] = metric_func(actuals, predictions)
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to calculate overall metric {i}: {e}")
                    overall_metrics[f'overall_metric_{i}'] = float('nan')

        validation_time = time.time() - start_time

        result = {
            'fold_results': fold_results,
            'overall_metrics': overall_metrics,
            'n_splits': n_splits,
            'shuffle': shuffle,
            'random_state': random_state,
            'total_samples': len(data),
            'validation_time': validation_time,
            'success': len(fold_results) > 0 and any(r.get('success', True) for r in fold_results)
        }

        self.logger.info(f"✅ K-fold cross-validation completed in {validation_time:.2f}s")
        return result

    def temporal_cross_validation(
        self,
        data: pd.DataFrame,
        model_function: Callable,
        target_column: str,
        time_column: str,
        n_splits: int = 5,
        test_ratio: float = 0.2,
        metric_functions: Optional[List[Callable]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform temporal cross-validation preserving time order.

        Args:
            data: Time series data
            model_function: Function that returns a fitted model
            target_column: Name of target column
            time_column: Name of time column
            n_splits: Number of validation splits
            test_ratio: Ratio of data for testing
            metric_functions: List of metric functions to evaluate

        Returns:
            Dictionary containing validation results
        """
        self.logger.info(f"🔍 Starting temporal cross-validation with {n_splits} splits")

        start_time = time.time()

        if metric_functions is None:
            metric_functions = [
                lambda y_true, y_pred: mean_squared_error(y_true, y_pred),
                lambda y_true, y_pred: mean_absolute_error(y_true, y_pred),
                lambda y_true, y_pred: r2_score(y_true, y_pred)
            ]

        # Sort data by time
        data_sorted = data.sort_values(time_column).reset_index(drop=True)

        # Initialize results storage
        fold_results = []
        predictions = []
        actuals = []

        total_samples = len(data_sorted)
        test_size = int(total_samples * test_ratio)

        for fold in range(n_splits):
            self.logger.debug(f"📈 Processing fold {fold + 1}/{n_splits}")

            # Calculate split indices
            split_point = total_samples - (n_splits - fold) * test_size

            # Ensure we don't go below reasonable training size
            min_train_size = int(total_samples * 0.1)  # At least 10% for training
            if split_point < min_train_size:
                split_point = min_train_size

            # Split data
            train_data = data_sorted.iloc[:split_point]
            test_data = data_sorted.iloc[split_point:]

            if len(train_data) == 0 or len(test_data) == 0:
                self.logger.warning(f"⚠️ Empty train or test set in fold {fold + 1}, skipping")
                continue

            # Prepare features and target
            X_train = train_data.drop(columns=[target_column])
            y_train = train_data[target_column]
            X_test = test_data.drop(columns=[target_column])
            y_test = test_data[target_column]

            try:
                # Train model
                model = model_function(X_train, y_train, **kwargs)

                # Make predictions
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_test)
                else:
                    # Assume model is a function
                    y_pred = model(X_test)

                # Calculate metrics
                fold_metrics = {}
                for i, metric_func in enumerate(metric_functions):
                    try:
                        metric_value = metric_func(y_test, y_pred)
                        fold_metrics[f'metric_{i}'] = metric_value
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to calculate metric {i}: {e}")
                        fold_metrics[f'metric_{i}'] = float('nan')

                # Store fold results
                fold_result = {
                    'fold': fold + 1,
                    'train_size': len(train_data),
                    'test_size': len(test_data),
                    'metrics': fold_metrics,
                    'predictions': y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred),
                    'actuals': y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test)
                }
                fold_results.append(fold_result)

                # Store predictions and actuals for overall metrics
                predictions.extend(y_pred)
                actuals.extend(y_test)

            except Exception as e:
                self.logger.error(f"❌ Fold {fold + 1} failed: {e}")
                fold_result = {
                    'fold': fold + 1,
                    'train_size': len(train_data),
                    'test_size': len(test_data),
                    'error': str(e),
                    'success': False
                }
                fold_results.append(fold_result)

        # Calculate overall metrics
        overall_metrics = {}
        if predictions and actuals:
            for i, metric_func in enumerate(metric_functions):
                try:
                    overall_metrics[f'overall_metric_{i}'] = metric_func(actuals, predictions)
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to calculate overall metric {i}: {e}")
                    overall_metrics[f'overall_metric_{i}'] = float('nan')

        validation_time = time.time() - start_time

        result = {
            'fold_results': fold_results,
            'overall_metrics': overall_metrics,
            'n_splits': n_splits,
            'test_ratio': test_ratio,
            'total_samples': total_samples,
            'validation_time': validation_time,
            'success': len(fold_results) > 0 and any(r.get('success', True) for r in fold_results)
        }

        self.logger.info(f"✅ Temporal cross-validation completed in {validation_time:.2f}s")
        return result


# Global instance for easy access
_cv_instance = None

def get_cross_validation_utilities() -> CrossValidationUtilities:
    """Get global cross-validation utilities instance."""
    global _cv_instance
    if _cv_instance is None:
        _cv_instance = CrossValidationUtilities()
    return _cv_instance

# Export key classes and functions
__all__ = ['CrossValidationUtilities', 'get_cross_validation_utilities']
