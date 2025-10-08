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
    """Feature selection framework with memory management.

    NOTE: Core selection logic is now centralized in
    `src/utils/feature_selection/framework.py`. This class acts as a thin
    adapter to preserve imports and backward compatibility while delegating
    to the central bank.
    """

    def __init__(self):
        """Initialize feature selection framework."""
        self.logger = logger.getChild('FeatureSelectionFramework')
        self.logger.info("🚀 Initializing FeatureSelectionFramework (delegating to central bank)")
        try:
            # Lazy import to avoid cycles
            from src.feature_selection.core.framework import get_feature_selection_framework as _get_bank_framework
            self._bank_framework = _get_bank_framework()
        except Exception as e:
            self.logger.warning(f"⚠️ Central bank framework unavailable, local utilities will be used: {e}")
            self._bank_framework = None

    def _validate_data_quality(self, X: np.ndarray, y: np.ndarray, context: str = "feature_selection") -> Dict[str, Any]:
        """Deprecated local validation. Central bank handles data checks. Kept for compatibility."""
        return {
            'is_valid': True,
            'issues': [],
            'data_shape': getattr(X, 'shape', None),
            'target_shape': getattr(y, 'shape', None),
            'needs_preprocessing': False
        }

    def _preprocess_data_for_ml(self, X: np.ndarray, y: np.ndarray, validation_results: Dict[str, Any], context: str = "feature_selection") -> np.ndarray:
        """Deprecated local preprocessing. Central bank handles preprocessing."""
        return X

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
            # Delegate to central bank if available
            if self._bank_framework is not None:
                from src.feature_selection.core.framework import select_features as bank_select

                # Map args to central API
                is_classification = task_type == 'classification'
                bank_result = bank_select(
                    pd.DataFrame(X, columns=feature_names) if not hasattr(X, 'values') else X,
                    y,
                    method=method,
                    max_features=max_features or k,
                    is_classification=is_classification,
                    feature_names=feature_names,
                    framework_config=None,
                )

                # Project bank result to legacy shape
                results.update({
                    'selected_features': bank_result.get('selected_features', []),
                    'n_selected_features': len(bank_result.get('selected_features', [])),
                    'selection_ratio': (len(bank_result.get('selected_features', [])) / len(feature_names)) if feature_names else None,
                    'feature_scores': bank_result.get('final_scores', bank_result.get('feature_scores', {})),
                    'feature_ranking': [],
                    'success': bank_result.get('success', True),
                    'method': bank_result.get('method', method),
                })

                # Optional analysis
                try:
                    if results['selected_features'] and not hasattr(X, 'values'):
                        # If X was ndarray, build indices
                        idxs = [feature_names.index(f) for f in results['selected_features'] if f in feature_names]
                    else:
                        idxs = [feature_names.index(f) for f in results['selected_features']]
                    results['feature_importance_analysis'] = self._analyze_feature_importance(
                        X if not hasattr(X, 'values') else X.values, y, idxs, task_type
                    )
                except Exception as e:
                    logger.error(f"❌ Critical error: Feature importance analysis failed: {e}")
                    raise ValueError(f"Feature selection analysis failed: {e}")
            else:
                # Fallback to local strategies for compatibility
                if method == 'auto':
                    selected_method = self._choose_optimal_method(X, y, task_type)
                    self.logger.info(f"📊 Auto-selected method: {selected_method}")
                    method = selected_method

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

                results['selected_features'] = [feature_names[i] for i in selected_features]
                results['feature_scores'] = {feature_names[i]: float(scores[i]) for i in range(len(scores))}
                results['feature_ranking'] = [(feature_names[i], float(ranking[i])) for i in range(len(ranking))]
                results['n_selected_features'] = len(selected_features)
                results['selection_ratio'] = len(selected_features) / len(feature_names)
                results['feature_importance_analysis'] = self._analyze_feature_importance(
                    X, y, selected_features, task_type
                )
                results['success'] = True
        except Exception as e:
            self.logger.error(f"❌ Feature selection failed: {e}")
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
            results.update({'error': str(e), 'error_diagnostics': error_diagnostics, 'success': False})

        results['selection_time'] = time.time() - start_time

        self.logger.info(f"✅ Feature selection completed in {results['selection_time']:.3f}s")
        return results

    def _choose_optimal_method(self, X: np.ndarray, y: np.ndarray, task_type: str) -> str:
        """Deprecated: selection strategy handled by central bank. Kept for compatibility."""
        return 'comprehensive'

    def _filter_based_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str,
        k: int,
        **kwargs
    ) -> Tuple[List[int], np.ndarray, np.ndarray]:
        """Deprecated local path. Use central bank."""
        raise RuntimeError("filter_based_selection is deprecated; use central framework")

    def _wrapper_based_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str,
        k: int,
        **kwargs
    ) -> Tuple[List[int], np.ndarray, np.ndarray]:
        """Deprecated local path. Use central bank."""
        raise RuntimeError("wrapper_based_selection is deprecated; use central framework")

    def _embedded_based_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str,
        k: int,
        **kwargs
    ) -> Tuple[List[int], np.ndarray, np.ndarray]:
        """Deprecated local path. Use central bank."""
        raise RuntimeError("embedded_based_selection is deprecated; use central framework")

    def _hybrid_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str,
        k: int,
        **kwargs
    ) -> Tuple[List[int], np.ndarray, np.ndarray]:
        """Deprecated local path. Use central bank."""
        raise RuntimeError("hybrid_selection is deprecated; use central framework")

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
        """Deprecated iterative sweep. Use central comprehensive pipeline instead."""
        return {
            'results': [],
            'best_result': None,
            'optimal_n_features': None,
            'best_score': None,
            'iteration_time': 0.0,
            'success': False,
            'error': 'iterative_feature_selection is deprecated; use central framework',
        }

    def _evaluate_feature_subset(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series, List],
        selected_features: List[str],
        evaluation_metric: Callable,
        task_type: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Deprecated local evaluation. Use caller-provided evaluation flow."""
        return {'success': False, 'error': 'deprecated'}


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
