"""
Comprehensive Validation Framework

This module provides comprehensive validation for feature selection methods
including cross-validation, regression testing, and performance metrics.
"""

import logging
import time
import json
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Import project utilities
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_performance, tprint_debug
from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager, HardwareConfig

logger = logging.getLogger(__name__)

@dataclass
class ValidationConfig:
    """Configuration for validation framework."""
    # Cross-validation settings
    cv_folds: int = 5
    cv_strategy: str = 'kfold'  # 'kfold', 'timeseries', 'stratified'
    test_size: float = 0.2
    random_state: int = 42

    # Regression testing settings
    enable_regression_testing: bool = True
    regression_threshold: float = 0.1  # 10% performance degradation threshold
    reference_results_path: Optional[str] = None

    # Performance metrics
    enable_performance_metrics: bool = True
    enable_stability_metrics: bool = True
    enable_interpretability_metrics: bool = True

    # Hardware optimization
    enable_hardware_optimization: bool = True
    n_jobs: int = -1

    # Validation models
    validation_models: List[str] = field(default_factory=lambda: ['linear', 'random_forest'])

    # Stability settings
    stability_n_repeats: int = 10
    stability_threshold: float = 0.8

class ValidationMetrics:
    """Comprehensive validation metrics for feature selection."""

    def __init__(self):
        """Initialize validation metrics."""
        self.metrics = {}
        self.logger = logger.getChild('ValidationMetrics')

    def calculate_selection_metrics(self, X: np.ndarray, y: np.ndarray,
                                  selected_features: List[str],
                                  feature_names: List[str]) -> Dict[str, Any]:
        """Calculate comprehensive selection metrics."""
        try:
            # Get selected indices
            selected_indices = [feature_names.index(f) for f in selected_features if f in feature_names]
            X_selected = X[:, selected_indices]

            # Basic metrics
            metrics = {
                'n_selected': len(selected_features),
                'n_total': len(feature_names),
                'selection_ratio': len(selected_features) / len(feature_names),
                'reduction_ratio': 1 - (len(selected_features) / len(feature_names))
            }

            # Feature quality metrics
            quality_metrics = self._calculate_quality_metrics(X_selected, y)
            metrics.update(quality_metrics)

            # Stability metrics
            stability_metrics = self._calculate_stability_metrics(X, y, selected_indices)
            metrics.update(stability_metrics)

            # Interpretability metrics
            interpretability_metrics = self._calculate_interpretability_metrics(X_selected, y)
            metrics.update(interpretability_metrics)

            return {
                'success': True,
                'metrics': metrics
            }

        except Exception as e:
            self.logger.error(f"Metrics calculation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _calculate_quality_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Calculate feature quality metrics."""
        try:
            # Variance metrics
            feature_variances = np.var(X, axis=0)
            variance_metrics = {
                'mean_variance': float(np.mean(feature_variances)),
                'variance_std': float(np.std(feature_variances)),
                'low_variance_features': int(np.sum(feature_variances < 0.01))
            }

            # Correlation metrics
            if X.shape[1] > 1:
                corr_matrix = np.corrcoef(X.T)
                high_corr_pairs = np.sum(np.abs(corr_matrix) > 0.95) - X.shape[1]  # Exclude diagonal
                correlation_metrics = {
                    'mean_correlation': float(np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))),
                    'high_correlation_pairs': int(high_corr_pairs)
                }
            else:
                correlation_metrics = {
                    'mean_correlation': 0.0,
                    'high_correlation_pairs': 0
                }

            # Information content
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(X, y, random_state=42)
            information_metrics = {
                'mean_mutual_info': float(np.mean(mi_scores)),
                'mutual_info_std': float(np.std(mi_scores)),
                'high_mi_features': int(np.sum(mi_scores > 0.1))
            }

            return {
                **variance_metrics,
                **correlation_metrics,
                **information_metrics
            }

        except Exception as e:
            self.logger.warning(f"Quality metrics calculation failed: {e}")
            return {}

    def _calculate_stability_metrics(self, X: np.ndarray, y: np.ndarray,
                                   selected_indices: List[int]) -> Dict[str, Any]:
        """Calculate stability metrics."""
        try:
            # Bootstrap stability
            n_bootstrap = 10
            stability_scores = []

            for i in range(n_bootstrap):
                # Bootstrap sample
                n_samples = X.shape[0]
                bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_bootstrap = X[bootstrap_indices]
                y_bootstrap = y[bootstrap_indices]

                # Calculate feature importance for bootstrap
                from sklearn.ensemble import RandomForestRegressor
                rf = RandomForestRegressor(n_estimators=50, random_state=42)
                rf.fit(X_bootstrap, y_bootstrap)
                importance = rf.feature_importances_

                # Check if selected features are in top features
                top_features = np.argsort(importance)[-len(selected_indices):]
                stability = len(set(selected_indices) & set(top_features)) / len(selected_indices)
                stability_scores.append(stability)

            return {
                'stability_mean': float(np.mean(stability_scores)),
                'stability_std': float(np.std(stability_scores)),
                'stability_min': float(np.min(stability_scores)),
                'stability_max': float(np.max(stability_scores))
            }

        except Exception as e:
            self.logger.warning(f"Stability metrics calculation failed: {e}")
            return {}

    def _calculate_interpretability_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Calculate interpretability metrics."""
        try:
            # Feature count (fewer features = more interpretable)
            interpretability_metrics = {
                'feature_count': X.shape[1],
                'interpretability_score': max(0, 1 - (X.shape[1] / 100))  # Normalized score
            }

            # Feature independence (lower correlation = more interpretable)
            if X.shape[1] > 1:
                corr_matrix = np.corrcoef(X.T)
                mean_correlation = np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
                interpretability_metrics['feature_independence'] = float(1 - mean_correlation)
            else:
                interpretability_metrics['feature_independence'] = 1.0

            return interpretability_metrics

        except Exception as e:
            self.logger.warning(f"Interpretability metrics calculation failed: {e}")
            return {}

class CrossValidationFramework:
    """Cross-validation framework for feature selection validation."""

    def __init__(self, config: Optional[ValidationConfig] = None):
        """Initialize cross-validation framework."""
        self.config = config or ValidationConfig()
        self.logger = logger.getChild('CrossValidationFramework')

        # Initialize hardware optimization
        if self.config.enable_hardware_optimization:
            self.cpu_optimizer = M1CPUOptimizer()
            hw_config = HardwareConfig(
                cpu_optimization_level='balanced',
                enable_adaptive_optimization=True
            )
            self.hardware_manager = UnifiedHardwareManager(hw_config)
        else:
            self.cpu_optimizer = None
            self.hardware_manager = None

        # Initialize metrics calculator
        self.metrics_calculator = ValidationMetrics()

        tprint_success("🔧 CrossValidationFramework initialized")

    def validate_selection_method(self, X: np.ndarray, y: np.ndarray,
                                selection_func: Callable,
                                feature_names: Optional[List[str]] = None,
                                **kwargs) -> Dict[str, Any]:
        """Validate a feature selection method using cross-validation."""
        tprint_info(f"🔧 Validating selection method with {self.config.cv_folds}-fold CV")

        start_time = time.time()

        try:
            # Prepare feature names
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]

            # Create CV splits
            cv_splits = self._create_cv_splits(X, y)

            # Validate on each fold
            fold_results = []
            for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
                tprint_debug(f"🔧 Validating fold {fold_idx + 1}/{self.config.cv_folds}")

                # Split data
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Run selection on training data
                selection_result = selection_func(X_train, y_train, **kwargs)

                if not selection_result.get('success', False):
                    tprint_warning(f"⚠️ Selection failed on fold {fold_idx + 1}")
                    continue

                # Get selected features
                selected_features = selection_result['selected_features']
                selected_indices = [feature_names.index(f) for f in selected_features if f in feature_names]

                # Evaluate on test data
                fold_metrics = self._evaluate_fold(
                    X_test, y_test, selected_indices, feature_names
                )

                fold_metrics['fold'] = fold_idx
                fold_metrics['n_selected'] = len(selected_features)
                fold_results.append(fold_metrics)

            # Aggregate results
            aggregated_results = self._aggregate_fold_results(fold_results)

            end_time = time.time()
            execution_time = end_time - start_time

            result = {
                'success': True,
                'cv_results': fold_results,
                'aggregated_results': aggregated_results,
                'n_folds': len(fold_results),
                'execution_time': execution_time,
                'validation_summary': self._create_validation_summary(aggregated_results)
            }

            tprint_success(f"✅ Cross-validation completed in {execution_time:.3f}s")

            return result

        except Exception as e:
            self.logger.error(f"Cross-validation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }

    def _create_cv_splits(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create cross-validation splits."""
        if self.config.cv_strategy == 'timeseries':
            cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        elif self.config.cv_strategy == 'stratified':
            # Determine if classification
            is_classification = len(np.unique(y)) < 10 and np.all(y == y.astype(int))
            if is_classification:
                cv = StratifiedKFold(n_splits=self.config.cv_folds, random_state=self.config.random_state)
            else:
                cv = KFold(n_splits=self.config.cv_folds, random_state=self.config.random_state)
        else:  # kfold
            cv = KFold(n_splits=self.config.cv_folds, random_state=self.config.random_state)

        return list(cv.split(X, y))

    def _evaluate_fold(self, X_test: np.ndarray, y_test: np.ndarray,
                      selected_indices: List[int], feature_names: List[str]) -> Dict[str, Any]:
        """Evaluate a single fold."""
        try:
            # Get selected features
            X_selected = X_test[:, selected_indices]

            # Calculate metrics
            metrics = self.metrics_calculator.calculate_selection_metrics(
                X_test, y_test, [feature_names[i] for i in selected_indices], feature_names
            )

            # Model performance metrics
            model_metrics = self._calculate_model_performance(X_selected, y_test)

            return {
                **metrics['metrics'],
                **model_metrics
            }

        except Exception as e:
            self.logger.warning(f"Fold evaluation failed: {e}")
            return {'error': str(e)}

    def _calculate_model_performance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Calculate model performance metrics."""
        try:
            # Determine if classification or regression
            is_classification = len(np.unique(y)) < 10 and np.all(y == y.astype(int))

            if is_classification:
                # Classification metrics
                model = LogisticRegression(random_state=42, max_iter=1000)
                model.fit(X, y)
                y_pred = model.predict(X)

                accuracy = accuracy_score(y, y_pred)

                return {
                    'model_accuracy': float(accuracy),
                    'model_type': 'classification'
                }
            else:
                # Regression metrics
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)

                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)

                return {
                    'model_mse': float(mse),
                    'model_r2': float(r2),
                    'model_type': 'regression'
                }

        except Exception as e:
            self.logger.warning(f"Model performance calculation failed: {e}")
            return {}

    def _aggregate_fold_results(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across folds."""
        if not fold_results:
            return {}

        # Get all metric names
        all_metrics = set()
        for result in fold_results:
            all_metrics.update(result.keys())

        # Calculate statistics for each metric
        aggregated = {}
        for metric in all_metrics:
            if metric in ['fold', 'error']:
                continue

            values = [result[metric] for result in fold_results if metric in result and isinstance(result[metric], (int, float))]

            if values:
                aggregated[f'{metric}_mean'] = float(np.mean(values))
                aggregated[f'{metric}_std'] = float(np.std(values))
                aggregated[f'{metric}_min'] = float(np.min(values))
                aggregated[f'{metric}_max'] = float(np.max(values))

        return aggregated

    def _create_validation_summary(self, aggregated_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create validation summary."""
        summary = {
            'overall_quality': 'good' if aggregated_results.get('model_r2_mean', 0) > 0.7 else 'fair',
            'stability': 'stable' if aggregated_results.get('stability_mean_mean', 0) > 0.8 else 'unstable',
            'interpretability': 'high' if aggregated_results.get('feature_count_mean', 0) < 20 else 'medium'
        }

        return summary

class RegressionTestFramework:
    """Regression testing framework for feature selection methods."""

    def __init__(self, config: Optional[ValidationConfig] = None):
        """Initialize regression test framework."""
        self.config = config or ValidationConfig()
        self.logger = logger.getChild('RegressionTestFramework')

        # Set up reference results path
        if self.config.reference_results_path is None:
            self.config.reference_results_path = "data_cache/feature_selection/reference_results.json"

        self.reference_path = Path(self.config.reference_results_path)
        self.reference_path.parent.mkdir(parents=True, exist_ok=True)

        tprint_success("🔧 RegressionTestFramework initialized")

    def run_regression_test(self, X: np.ndarray, y: np.ndarray,
                          selection_func: Callable,
                          test_name: str,
                          feature_names: Optional[List[str]] = None,
                          **kwargs) -> Dict[str, Any]:
        """Run regression test for feature selection method."""
        tprint_info(f"🔧 Running regression test: {test_name}")

        try:
            # Run current selection
            current_result = selection_func(X, y, **kwargs)

            if not current_result.get('success', False):
                return {
                    'success': False,
                    'error': 'Selection method failed',
                    'test_name': test_name
                }

            # Load reference results
            reference_results = self._load_reference_results()

            if test_name in reference_results:
                # Compare with reference
                reference_result = reference_results[test_name]
                comparison_result = self._compare_results(current_result, reference_result)

                # Check for regression
                regression_detected = self._detect_regression(comparison_result)

                result = {
                    'success': True,
                    'test_name': test_name,
                    'current_result': current_result,
                    'reference_result': reference_result,
                    'comparison': comparison_result,
                    'regression_detected': regression_detected,
                    'regression_threshold': self.config.regression_threshold
                }

                if regression_detected:
                    tprint_warning(f"⚠️ Regression detected in {test_name}")
                else:
                    tprint_success(f"✅ No regression detected in {test_name}")

            else:
                # No reference available, save current result
                self._save_reference_result(test_name, current_result)
                result = {
                    'success': True,
                    'test_name': test_name,
                    'current_result': current_result,
                    'reference_saved': True,
                    'message': 'Reference result saved for future comparison'
                }
                tprint_info(f"💾 Reference result saved for {test_name}")

            return result

        except Exception as e:
            self.logger.error(f"Regression test failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'test_name': test_name
            }

    def _load_reference_results(self) -> Dict[str, Any]:
        """Load reference results from file."""
        try:
            if self.reference_path.exists():
                with open(self.reference_path, 'r') as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            self.logger.warning(f"Failed to load reference results: {e}")
            return {}

    def _save_reference_result(self, test_name: str, result: Dict[str, Any]) -> None:
        """Save reference result to file."""
        try:
            reference_results = self._load_reference_results()
            reference_results[test_name] = result

            with open(self.reference_path, 'w') as f:
                json.dump(reference_results, f, indent=2)

        except Exception as e:
            self.logger.warning(f"Failed to save reference result: {e}")

    def _compare_results(self, current: Dict[str, Any], reference: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current and reference results."""
        comparison = {}

        # Compare key metrics
        key_metrics = ['n_selected', 'n_total', 'selection_ratio']

        for metric in key_metrics:
            if metric in current and metric in reference:
                current_val = current[metric]
                reference_val = reference[metric]

                if reference_val != 0:
                    relative_change = (current_val - reference_val) / reference_val
                else:
                    relative_change = float('inf') if current_val != 0 else 0.0

                comparison[f'{metric}_change'] = relative_change
                comparison[f'{metric}_current'] = current_val
                comparison[f'{metric}_reference'] = reference_val

        # Compare selected features
        current_features = set(current.get('selected_features', []))
        reference_features = set(reference.get('selected_features', []))

        if reference_features:
            feature_overlap = len(current_features & reference_features) / len(reference_features)
            comparison['feature_overlap'] = feature_overlap
        else:
            comparison['feature_overlap'] = 0.0

        return comparison

    def _detect_regression(self, comparison: Dict[str, Any]) -> bool:
        """Detect if regression has occurred."""
        # Check for significant changes in key metrics
        key_changes = ['n_selected_change', 'selection_ratio_change']

        for change_key in key_changes:
            if change_key in comparison:
                change = abs(comparison[change_key])
                if change > self.config.regression_threshold:
                    return True

        # Check feature overlap
        if 'feature_overlap' in comparison:
            if comparison['feature_overlap'] < (1 - self.config.regression_threshold):
                return True

        return False

class FeatureSelectionValidator:
    """Main validator combining all validation methods."""

    def __init__(self, config: Optional[ValidationConfig] = None):
        """Initialize feature selection validator."""
        self.config = config or ValidationConfig()
        self.logger = logger.getChild('FeatureSelectionValidator')

        # Initialize components
        self.cv_framework = CrossValidationFramework(self.config)
        self.regression_framework = RegressionTestFramework(self.config)
        self.metrics_calculator = ValidationMetrics()

        tprint_success("🔧 FeatureSelectionValidator initialized")

    def validate_selection_method(self, X: np.ndarray, y: np.ndarray,
                                selection_func: Callable,
                                test_name: str = "default_test",
                                feature_names: Optional[List[str]] = None,
                                **kwargs) -> Dict[str, Any]:
        """Comprehensive validation of feature selection method."""
        tprint_info(f"🔧 Comprehensive validation: {test_name}")

        start_time = time.time()

        try:
            validation_results = {
                'test_name': test_name,
                'timestamp': time.time(),
                'data_shape': X.shape
            }

            # Cross-validation
            tprint_info("🔧 Running cross-validation...")
            cv_result = self.cv_framework.validate_selection_method(
                X, y, selection_func, feature_names, **kwargs
            )
            validation_results['cross_validation'] = cv_result

            # Regression testing
            if self.config.enable_regression_testing:
                tprint_info("🔧 Running regression test...")
                regression_result = self.regression_framework.run_regression_test(
                    X, y, selection_func, test_name, feature_names, **kwargs
                )
                validation_results['regression_test'] = regression_result

            # Overall validation summary
            validation_results['overall_success'] = (
                cv_result.get('success', False) and
                (not self.config.enable_regression_testing or
                 regression_result.get('success', False))
            )

            end_time = time.time()
            validation_results['total_execution_time'] = end_time - start_time

            if validation_results['overall_success']:
                tprint_success(f"✅ Validation completed successfully in {validation_results['total_execution_time']:.3f}s")
            else:
                tprint_warning(f"⚠️ Validation completed with issues in {validation_results['total_execution_time']:.3f}s")

            return validation_results

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'test_name': test_name,
                'execution_time': time.time() - start_time
            }

def create_validation_framework(config: Optional[ValidationConfig] = None) -> FeatureSelectionValidator:
    """Create a feature selection validator."""
    return FeatureSelectionValidator(config)
