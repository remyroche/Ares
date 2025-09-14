"""
Feature Selection Framework

This module provides comprehensive feature selection utilities with memory-aware operations.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest, f_regression, f_classif, mutual_info_regression, mutual_info_classif,
    RFE, RFECV, SelectFromModel
)
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score

logger = logging.getLogger(__name__)

class FeatureSelectionFramework:
    """Feature selection framework with memory management."""

    def __init__(self):
        """Initialize feature selection framework."""
        self.logger = logger.getChild('FeatureSelectionFramework')
        self.logger.info("🚀 Initializing FeatureSelectionFramework")

    def _validate_data_quality(self, X: np.ndarray, y: np.ndarray, context: str = "feature_selection") -> Dict[str, Any]:
        """
        Validate data quality and return diagnostics.

        Args:
            X: Feature matrix
            y: Target vector
            context: Context for logging

        Returns:
            Dictionary with validation results and any preprocessing needed
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'data_shape': X.shape,
            'target_shape': y.shape,
            'needs_preprocessing': False
        }

        # Check for NaN values in features
        nan_mask_X = np.isnan(X)
        nan_count_X = np.sum(nan_mask_X)
        if nan_count_X > 0:
            validation_results['issues'].append(f"Found {nan_count_X} NaN values in feature matrix")
            validation_results['features_have_nan'] = True
            validation_results['needs_preprocessing'] = True
            self.logger.warning(f"⚠️ {context}: Found {nan_count_X} NaN values in feature matrix ({(nan_count_X/X.size)*100:.2f}%)")

        # Check for infinity values in features
        inf_mask_X = np.isinf(X)
        inf_count_X = np.sum(inf_mask_X)
        if inf_count_X > 0:
            validation_results['issues'].append(f"Found {inf_count_X} infinity values in feature matrix")
            validation_results['features_have_inf'] = True
            validation_results['needs_preprocessing'] = True
            self.logger.warning(f"⚠️ {context}: Found {inf_count_X} infinity values in feature matrix ({(inf_count_X/X.size)*100:.2f}%)")

        # Check for NaN values in target
        nan_mask_y = np.isnan(y)
        nan_count_y = np.sum(nan_mask_y)
        if nan_count_y > 0:
            validation_results['issues'].append(f"Found {nan_count_y} NaN values in target variable")
            validation_results['target_has_nan'] = True
            validation_results['is_valid'] = False
            self.logger.error(f"❌ {context}: Found {nan_count_y} NaN values in target variable - cannot proceed")

        # Check for constant target values
        unique_y = np.unique(y)
        if len(unique_y) == 1:
            validation_results['issues'].append(f"All target values are identical ({unique_y[0]})")
            validation_results['constant_target'] = True
            validation_results['is_valid'] = False
            self.logger.error(f"❌ {context}: All target values are identical ({unique_y[0]}) - cannot perform meaningful feature selection")

        # Check target data type issues for classification
        if len(unique_y) <= 10:  # Likely classification
            try:
                y_int = y.astype(int)
                if not np.array_equal(y, y_int):
                    validation_results['issues'].append("Target values are not integers for classification task")
                    validation_results['target_dtype_issue'] = True
                    self.logger.warning(f"⚠️ {context}: Target values are not integers, switching to regression")
            except (ValueError, TypeError):
                validation_results['issues'].append("Target values cannot be converted to integers for classification")
                validation_results['target_dtype_issue'] = True
                self.logger.warning(f"⚠️ {context}: Target values cannot be converted to integers, treating as regression")

        # Overall validation status
        validation_results['is_valid'] = validation_results['is_valid'] and len(validation_results['issues']) == 0

        return validation_results

    def _preprocess_data_for_ml(self, X: np.ndarray, y: np.ndarray, validation_results: Dict[str, Any], context: str = "feature_selection") -> np.ndarray:
        """
        Preprocess data to handle NaN and infinity values.

        Args:
            X: Feature matrix
            y: Target vector (for reference, not modified)
            validation_results: Results from data validation
            context: Context for logging

        Returns:
            Preprocessed feature matrix
        """
        X_processed = X.copy()

        # Handle NaN values in features
        if validation_results.get('features_have_nan', False):
            nan_mask = np.isnan(X_processed)
            nan_count = np.sum(nan_mask)
            self.logger.info(f"🔧 {context}: Filling {nan_count} NaN values in features with column means")

            # Fill NaN values with column means
            for col in range(X_processed.shape[1]):
                col_data = X_processed[:, col]
                nan_indices = np.isnan(col_data)
                if np.any(nan_indices):
                    finite_mask = np.isfinite(col_data)
                    if np.any(finite_mask):
                        col_mean = np.mean(col_data[finite_mask])
                        X_processed[nan_indices, col] = col_mean
                    else:
                        X_processed[nan_indices, col] = 0.0

        # Handle infinity values in features (similar to existing wrapper method)
        if validation_results.get('features_have_inf', False):
            inf_mask = np.isinf(X_processed)
            inf_count = np.sum(inf_mask)
            self.logger.info(f"🔧 {context}: Handling {inf_count} infinity values in features")

            # Replace positive infinity
            pos_inf_mask = np.isposinf(X_processed)
            if np.any(pos_inf_mask):
                finite_mask = np.isfinite(X_processed)
                if np.any(finite_mask):
                    max_finite = np.max(X_processed[finite_mask])
                    X_processed[pos_inf_mask] = max(max_finite * 10, 1e10)
                else:
                    X_processed[pos_inf_mask] = 1e10

            # Replace negative infinity
            neg_inf_mask = np.isneginf(X_processed)
            if np.any(neg_inf_mask):
                finite_mask = np.isfinite(X_processed)
                if np.any(finite_mask):
                    min_finite = np.min(X_processed[finite_mask])
                    X_processed[neg_inf_mask] = min(min_finite * 10, -1e10)
                else:
                    X_processed[neg_inf_mask] = -1e10

        # Clip extremely large values
        max_float64 = 1e308
        min_float64 = -1e308
        X_processed = np.clip(X_processed, min_float64, max_float64)

        return X_processed

    def select_features(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, List],
        task_type: str = 'regression',
        method: str = 'auto',
        k: Optional[int] = None,
        max_features: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform feature selection using various methods.

        Args:
            X: Feature matrix
            y: Target vector
            task_type: Type of ML task ('regression' or 'classification')
            method: Feature selection method
            k: Number of features to select (for filter methods)
            max_features: Maximum number of features to consider

        Returns:
            Dictionary containing feature selection results
        """
        self.logger.info(f"🔍 Starting feature selection with method: {method}")

        start_time = time.time()

        # Convert to numpy arrays if needed
        if hasattr(X, 'values'):
            feature_names = X.columns.tolist()
            X = X.values
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        y = np.array(y)

        # Validate data quality before proceeding
        validation_results = self._validate_data_quality(X, y, "iterative_feature_selection")
        if not validation_results['is_valid']:
            self.logger.error(f"❌ Data validation failed for iterative selection: {validation_results['issues']}")
            return {
                'results': [],
                'best_result': None,
                'optimal_n_features': None,
                'best_score': None,
                'iteration_time': 0.0,
                'success': False,
                'error': f"Data validation failed: {', '.join(validation_results['issues'])}",
                'validation_issues': validation_results['issues']
            }

        # Preprocess data if needed
        if validation_results['needs_preprocessing']:
            X = self._preprocess_data_for_ml(X, y, validation_results, "iterative_feature_selection")
            self.logger.info("✅ Data preprocessing completed for iterative selection")

        # Determine optimal k if not specified
        if k is None:
            k = min(len(feature_names) // 2, 50)  # Select up to 50 features or half of available

        if max_features is None:
            max_features = len(feature_names)

        results = {
            'method': method,
            'task_type': task_type,
            'original_features': len(feature_names),
            'selected_features': [],
            'feature_scores': {},
            'feature_ranking': [],
            'selection_time': None,
            'success': False
        }

        try:
            if method == 'auto':
                # Automatically choose best method based on data characteristics
                selected_method = self._choose_optimal_method(X, y, task_type)
                self.logger.info(f"📊 Auto-selected method: {selected_method}")
                method = selected_method

            # Apply selected method
            if method == 'filter':
                selected_features, scores, ranking = self._filter_based_selection(
                    X, y, task_type, k, **kwargs
                )
            elif method == 'wrapper':
                selected_features, scores, ranking = self._wrapper_based_selection(
                    X, y, task_type, k, **kwargs
                )
            elif method == 'embedded':
                selected_features, scores, ranking = self._embedded_based_selection(
                    X, y, task_type, k, **kwargs
                )
            elif method == 'hybrid':
                selected_features, scores, ranking = self._hybrid_selection(
                    X, y, task_type, k, **kwargs
                )
            else:
                raise ValueError(f"Unknown feature selection method: {method}")

            # Update results
            results['selected_features'] = [feature_names[i] for i in selected_features]
            results['feature_scores'] = {feature_names[i]: float(scores[i]) for i in range(len(scores))}
            results['feature_ranking'] = [(feature_names[i], float(ranking[i])) for i in range(len(ranking))]
            results['n_selected_features'] = len(selected_features)
            results['selection_ratio'] = len(selected_features) / len(feature_names)

            # Calculate feature importance metrics
            results['feature_importance_analysis'] = self._analyze_feature_importance(
                X, y, selected_features, task_type
            )

            results['success'] = True

        except Exception as e:
            self.logger.error(f"❌ Feature selection failed: {e}")

            # Enhanced error diagnostics
            error_diagnostics = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'method': method,
                'task_type': task_type,
                'data_shape': X.shape if 'X' in locals() else None,
                'target_shape': y.shape if 'y' in locals() else None,
                'k_parameter': k,
                'max_features': max_features,
                'feature_names_count': len(feature_names) if 'feature_names' in locals() else None
            }

            # Try to provide more specific guidance
            if "Input contains NaN" in str(e):
                error_diagnostics['guidance'] = "Data contains NaN values that weren't properly handled"
            elif "Input contains infinity" in str(e):
                error_diagnostics['guidance'] = "Data contains infinity values that weren't properly handled"
            elif "constant" in str(e).lower():
                error_diagnostics['guidance'] = "Target variable appears to be constant - check target data"
            elif "classification" in str(e).lower():
                error_diagnostics['guidance'] = "Classification task failed - check target variable format"
            elif "regression" in str(e).lower():
                error_diagnostics['guidance'] = "Regression task failed - check target variable format"
            else:
                error_diagnostics['guidance'] = "General feature selection failure - check data quality and parameters"

            results.update({
                'error': str(e),
                'error_diagnostics': error_diagnostics,
                'success': False
            })

        results['selection_time'] = time.time() - start_time

        self.logger.info(f"✅ Feature selection completed in {results['selection_time']:.3f}s")
        return results

    def _choose_optimal_method(self, X: np.ndarray, y: np.ndarray, task_type: str) -> str:
        """Choose optimal feature selection method based on data characteristics."""
        n_features = X.shape[1]
        n_samples = X.shape[0]

        # For small datasets, prefer filter methods
        if n_samples < 1000:
            return 'filter'
        # For high-dimensional data, use embedded methods
        elif n_features > 1000:
            return 'embedded'
        # For moderate datasets, use hybrid approach
        else:
            return 'hybrid'

    def _filter_based_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str,
        k: int,
        **kwargs
    ) -> Tuple[List[int], np.ndarray, np.ndarray]:
        """Filter-based feature selection."""
        self.logger.debug("📊 Using filter-based selection")

        if task_type == 'regression':
            # Use f_regression for regression
            selector = SelectKBest(score_func=f_regression, k=k)
            selector.fit(X, y)
            scores = selector.scores_
        else:
            # Use f_classif for classification
            selector = SelectKBest(score_func=f_classif, k=k)
            selector.fit(X, y)
            scores = selector.scores_

        # Get selected features
        selected_features = selector.get_support(indices=True)

        # Create ranking (higher score = better rank)
        ranking = np.argsort(scores)[::-1]  # Sort in descending order

        return selected_features.tolist(), scores, ranking

    def _wrapper_based_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str,
        k: int,
        **kwargs
    ) -> Tuple[List[int], np.ndarray, np.ndarray]:
        """Wrapper-based feature selection using RFE."""
        self.logger.debug("📊 Using wrapper-based selection")

        # Preprocess data to handle infinity and large values
        X_processed = X.copy()

        # Handle infinity values
        inf_mask = np.isinf(X_processed)
        if np.any(inf_mask):
            self.logger.warning(f"⚠️ Found {np.sum(inf_mask)} infinity values in data for wrapper RFE, replacing with finite values")

            # Replace positive infinity
            pos_inf_mask = np.isposinf(X_processed)
            if np.any(pos_inf_mask):
                finite_mask = np.isfinite(X_processed)
                if np.any(finite_mask):
                    max_finite = np.max(X_processed[finite_mask])
                    X_processed[pos_inf_mask] = max(max_finite * 10, 1e10)
                else:
                    X_processed[pos_inf_mask] = 1e10

            # Replace negative infinity
            neg_inf_mask = np.isneginf(X_processed)
            if np.any(neg_inf_mask):
                finite_mask = np.isfinite(X_processed)
                if np.any(finite_mask):
                    min_finite = np.min(X_processed[finite_mask])
                    X_processed[neg_inf_mask] = min(min_finite * 10, -1e10)
                else:
                    X_processed[neg_inf_mask] = -1e10

        # Clip extremely large values
        max_float64 = 1e308
        min_float64 = -1e308
        X_processed = np.clip(X_processed, min_float64, max_float64)

        # Use processed data for RFE
        X = X_processed

        # Choose base estimator
        if task_type == 'regression':
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)

        # Use RFE for feature selection
        selector = RFE(estimator=estimator, n_features_to_select=k)
        selector.fit(X, y)

        # Get selected features
        selected_features = selector.get_support(indices=True)

        # Get feature rankings (lower rank = better)
        ranking = selector.ranking_

        # Create scores (inverse of ranking)
        scores = 1.0 / ranking

        return selected_features.tolist(), scores, ranking

    def _embedded_based_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str,
        k: int,
        **kwargs
    ) -> Tuple[List[int], np.ndarray, np.ndarray]:
        """Embedded-based feature selection using Lasso or tree-based methods."""
        self.logger.debug("📊 Using embedded-based selection")

        if task_type == 'regression':
            # Use Lasso for regression
            estimator = LassoCV(cv=5, random_state=42)
        else:
            # Use Random Forest for classification
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)

        # Use SelectFromModel
        selector = SelectFromModel(estimator=estimator, max_features=k)
        selector.fit(X, y)

        # Get selected features
        selected_features = selector.get_support(indices=True)

        # Get feature importances
        if hasattr(selector.estimator_, 'coef_'):
            # Lasso coefficients
            scores = np.abs(selector.estimator_.coef_)
        elif hasattr(selector.estimator_, 'feature_importances_'):
            # Tree-based feature importances
            scores = selector.estimator_.feature_importances_
        else:
            # Fallback: equal scores
            scores = np.ones(X.shape[1])

        # Create ranking
        ranking = np.argsort(scores)[::-1]

        return selected_features.tolist(), scores, ranking

    def _hybrid_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str,
        k: int,
        **kwargs
    ) -> Tuple[List[int], np.ndarray, np.ndarray]:
        """Hybrid feature selection combining multiple methods."""
        self.logger.debug("📊 Using hybrid selection")

        # Step 1: Filter-based pre-selection (select 2x desired features)
        pre_k = min(k * 2, X.shape[1])
        filter_selected, filter_scores, _ = self._filter_based_selection(
            X, y, task_type, pre_k
        )

        # Step 2: Wrapper-based refinement on pre-selected features
        X_filtered = X[:, filter_selected]
        wrapper_selected, wrapper_scores, wrapper_ranking = self._wrapper_based_selection(
            X_filtered, y, task_type, k
        )

        # Map back to original feature indices
        selected_features = [filter_selected[i] for i in wrapper_selected]

        # Combine scores
        combined_scores = np.zeros(X.shape[1])
        combined_scores[filter_selected] = filter_scores[filter_selected]
        combined_scores[selected_features] *= 2  # Boost scores for wrapper-selected features

        # Create ranking for all original features (lower values = better rank)
        ranking = np.full(X.shape[1], len(filter_selected) + 1, dtype=int)  # Default to worst rank
        # Map wrapper rankings back to original feature indices
        for i, filter_idx in enumerate(filter_selected):
            ranking[filter_idx] = wrapper_ranking[i]

        return selected_features, combined_scores, ranking

    def _analyze_feature_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        selected_features: List[int],
        task_type: str
    ) -> Dict[str, Any]:
        """Analyze importance of selected features."""
        analysis = {}

        try:
            # Calculate correlation with target
            correlations = []
            for idx in selected_features:
                corr = np.corrcoef(X[:, idx], y)[0, 1]
                correlations.append(abs(corr))

            analysis['mean_correlation'] = float(np.mean(correlations))
            analysis['max_correlation'] = float(np.max(correlations))
            analysis['correlation_std'] = float(np.std(correlations))

            # Calculate feature stability (if we have multiple runs)
            analysis['feature_stability'] = 'single_run'

            # Calculate redundancy (correlation between selected features)
            if len(selected_features) > 1:
                feature_corr_matrix = np.corrcoef(X[:, selected_features].T)
                analysis['mean_feature_correlation'] = float(np.mean(np.abs(feature_corr_matrix)))
                analysis['max_feature_correlation'] = float(np.max(np.abs(feature_corr_matrix)))

        except Exception as e:
            self.logger.warning(f"⚠️ Feature importance analysis failed: {e}")
            analysis['error'] = str(e)

        return analysis

    def iterative_feature_selection(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, List],
        task_type: str = 'regression',
        min_features: int = 5,
        max_features: int = 50,
        step_size: int = 5,
        evaluation_metric: Optional[Callable] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform iterative feature selection to find optimal number of features.

        Args:
            X: Feature matrix
            y: Target vector
            task_type: Type of ML task
            min_features: Minimum number of features to test
            max_features: Maximum number of features to test
            step_size: Step size for feature count
            evaluation_metric: Metric function for evaluation

        Returns:
            Dictionary containing iterative selection results
        """
        self.logger.info(f"🔍 Starting iterative feature selection ({min_features}-{max_features})")

        start_time = time.time()

        # Set default evaluation metric
        if evaluation_metric is None:
            if task_type == 'regression':
                evaluation_metric = lambda y_true, y_pred: r2_score(y_true, y_pred)
            else:
                evaluation_metric = lambda y_true, y_pred: accuracy_score(y_true, y_pred)

        results = []
        best_result = None
        best_score = float('-inf')

        # Test different numbers of features
        for n_features in range(min_features, max_features + 1, step_size):
            self.logger.debug(f"📊 Testing {n_features} features")

            try:
                # Perform feature selection
                selection_result = self.select_features(
                    X, y, task_type=task_type, k=n_features, **kwargs
                )

                if not selection_result['success']:
                    continue

                # Evaluate selected features
                evaluation_result = self._evaluate_feature_subset(
                    X, y, selection_result['selected_features'],
                    evaluation_metric, task_type, **kwargs
                )

                result = {
                    'n_features': n_features,
                    'selected_features': selection_result['selected_features'],
                    'selection_result': selection_result,
                    'evaluation_score': evaluation_result.get('score', float('nan')),
                    'evaluation_details': evaluation_result
                }
                results.append(result)

                # Track best result
                if evaluation_result.get('score', float('nan')) > best_score:
                    best_score = evaluation_result['score']
                    best_result = result

            except Exception as e:
                self.logger.warning(f"⚠️ Failed evaluation for {n_features} features: {e}")
                result = {
                    'n_features': n_features,
                    'error': str(e),
                    'success': False
                }
                results.append(result)

        iteration_time = time.time() - start_time

        final_result = {
            'results': results,
            'best_result': best_result,
            'optimal_n_features': best_result['n_features'] if best_result else None,
            'best_score': best_score if best_result else None,
            'iteration_time': iteration_time,
            'success': best_result is not None
        }

        self.logger.info(f"✅ Iterative feature selection completed in {iteration_time:.3f}s")
        return final_result

    def _evaluate_feature_subset(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, List],
        selected_features: List[str],
        evaluation_metric: Callable,
        task_type: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate a subset of features using cross-validation."""
        try:
            # Get feature indices
            if hasattr(X, 'columns'):
                feature_indices = [X.columns.get_loc(feat) for feat in selected_features]
                X_subset = X.iloc[:, feature_indices].values
            else:
                # Assume X is numpy array and selected_features contains indices
                feature_indices = selected_features
                X_subset = X[:, feature_indices]

            # Convert y to numpy array if needed
            y_array = np.array(y)

            # Validate subset data quality
            validation_results = self._validate_data_quality(X_subset, y_array, "feature_subset_evaluation")
            if not validation_results['is_valid']:
                self.logger.warning(f"⚠️ Feature subset validation failed: {validation_results['issues']}")
                return {
                    'error': f"Subset validation failed: {', '.join(validation_results['issues'])}",
                    'validation_issues': validation_results['issues'],
                    'success': False
                }

            # Preprocess subset data if needed
            if validation_results['needs_preprocessing']:
                X_subset = self._preprocess_data_for_ml(X_subset, y_array, validation_results, "feature_subset_evaluation")

            # Simple train-test split evaluation
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                X_subset, y_array, test_size=0.2, random_state=42
            )

            # Train a simple model
            if task_type == 'regression':
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
            else:
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(random_state=42, max_iter=1000)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate score
            score = evaluation_metric(y_test, y_pred)

            return {
                'score': float(score),
                'n_features': len(selected_features),
                'success': True
            }

        except Exception as e:
            self.logger.error(f"❌ Feature subset evaluation failed: {e}")

            # Enhanced error diagnostics for evaluation failures
            error_diagnostics = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'n_selected_features': len(selected_features) if 'selected_features' in locals() else None,
                'task_type': task_type,
                'subset_shape': X_subset.shape if 'X_subset' in locals() else None,
                'target_shape': y_array.shape if 'y_array' in locals() else None
            }

            # Try to provide more specific guidance
            if "Input contains NaN" in str(e):
                error_diagnostics['guidance'] = "Subset contains NaN values that weren't properly handled"
            elif "Input contains infinity" in str(e):
                error_diagnostics['guidance'] = "Subset contains infinity values that weren't properly handled"
            elif "constant" in str(e).lower():
                error_diagnostics['guidance'] = "Target variable appears to be constant in this subset"
            elif "convergence" in str(e).lower():
                error_diagnostics['guidance'] = "Model failed to converge - try different model parameters"
            else:
                error_diagnostics['guidance'] = "General evaluation failure - check subset data quality"

            return {
                'error': str(e),
                'error_diagnostics': error_diagnostics,
                'success': False
            }


# Global instance for easy access
_feature_selection_instance = None

def get_feature_selection_framework() -> FeatureSelectionFramework:
    """Get global feature selection framework instance."""
    global _feature_selection_instance
    if _feature_selection_instance is None:
        _feature_selection_instance = FeatureSelectionFramework()
    return _feature_selection_instance

# Export key classes and functions
__all__ = ['FeatureSelectionFramework', 'get_feature_selection_framework']
