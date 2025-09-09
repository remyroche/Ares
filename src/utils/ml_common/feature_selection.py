"""
Unified Feature Selection Framework

This module provides a comprehensive feature selection framework combining multiple
selection methods with stability analysis, correlation filtering, and ensemble approaches.

Key Features:
- mRMR (Minimum Redundancy Maximum Relevance) selection
- Stability-weighted feature selection
- Correlation-based filtering
- Recursive feature elimination
- Feature importance ranking
- Composite feature scoring
- Cross-validated feature selection

Built on existing utilities:
- Uses math_validation.py for safe mathematical operations
- Integrates with m1_gpu_utils.py for GPU acceleration
- Leverages common_operations.py for robust error handling
- Builds on existing feature selection patterns
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime
import logging
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import warnings

from ..math_validation import safe_divide, safe_log
from ..common_operations import create_fallback_logger
from ..m1_gpu_utils import M1GPUManager
from ..parallel_processing_optimizer import ParallelProcessor

logger = logging.getLogger(__name__)

try:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.feature_selection import RFE, RFECV
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from scipy.stats import pearsonr, spearmanr
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available - limited feature selection functionality")


class FeatureSelectionFramework:
    """Comprehensive feature selection framework with multiple methods and stability analysis."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize feature selection framework with configuration."""
        self.config = config or {}
        self.logger = logger.getChild('FeatureSelection')

        # Configuration defaults
        self.enable_gpu = self.config.get('enable_gpu', True)
        self.enable_parallel = self.config.get('enable_parallel', True)
        self.max_workers = self.config.get('max_workers', 4)
        self.memory_threshold = self.config.get('memory_threshold', 0.8)
        self.random_state = self.config.get('random_state', 42)

        # Initialize utilities
        self.gpu_manager = M1GPUManager() if self.enable_gpu else None
        self.parallel_processor = ParallelProcessor() if self.enable_parallel else None

        # Method configurations
        self.method_configs = {
            'mrmr': {
                'relevance_method': 'mutual_info',
                'redundancy_method': 'correlation',
                'n_neighbors': 3
            },
            'importance': {
                'n_estimators': 100,
                'max_depth': 10,
                'bootstrap': True
            },
            'rfe': {
                'step': 0.1,
                'cv': 3,
                'scoring': 'accuracy'
            },
            'stability': {
                'n_bootstraps': 50,
                'bootstrap_fraction': 0.8,
                'stability_threshold': 0.6
            }
        }

        # Update with user config
        if 'method_configs' in self.config:
            self.method_configs.update(self.config['method_configs'])

    def mrmr_selection(self, X: np.ndarray, y: np.ndarray,
                      feature_names: List[str], n_features: int,
                      relevance_method: str = 'mutual_info',
                      redundancy_method: str = 'correlation') -> Dict[str, Any]:
        """
        Perform mRMR (Minimum Redundancy Maximum Relevance) feature selection.

        Args:
            X: Feature matrix
            y: Target array
            feature_names: List of feature names
            n_features: Number of features to select
            relevance_method: Method for relevance calculation ('mutual_info', 'correlation', 'importance')
            redundancy_method: Method for redundancy calculation ('correlation', 'mutual_info')

        Returns:
            Dictionary with selected features and scores
        """
        try:
            self.logger.info(f"🔍 Starting mRMR selection for {n_features} features")

            mrmr_results = {
                'selected_features': [],
                'feature_scores': {},
                'relevance_scores': {},
                'redundancy_scores': {},
                'mrmr_scores': {},
                'selection_metadata': {
                    'method': 'mrmr',
                    'relevance_method': relevance_method,
                    'redundancy_method': redundancy_method,
                    'n_features_requested': n_features
                }
            }

            # Calculate relevance scores
            relevance_scores = self._calculate_relevance_scores(X, y, feature_names, relevance_method)
            mrmr_results['relevance_scores'] = relevance_scores

            # mRMR algorithm
            selected_indices = []
            remaining_indices = list(range(len(feature_names)))

            # Start with most relevant feature
            if relevance_scores:
                try:
                    best_feature_name = max(relevance_scores.items(), key=lambda x: x[1])[0]
                    # Ensure best_feature_name is a string and exists in feature_names
                    if isinstance(best_feature_name, str) and best_feature_name in feature_names:
                        best_feature_idx = feature_names.index(best_feature_name)
                        selected_indices.append(best_feature_idx)
                        remaining_indices.remove(best_feature_idx)

                        mrmr_results['selected_features'].append(feature_names[best_feature_idx])
                        mrmr_results['mrmr_scores'][feature_names[best_feature_idx]] = relevance_scores[best_feature_name]
                    else:
                        self.logger.warning(f"⚠️ Invalid feature name in relevance scores: {best_feature_name}")
                        # Fallback: select first feature
                        if feature_names:
                            selected_indices.append(0)
                            remaining_indices.remove(0)
                            mrmr_results['selected_features'].append(feature_names[0])
                            mrmr_results['mrmr_scores'][feature_names[0]] = 0.0
                except (ValueError, KeyError, TypeError) as e:
                    self.logger.warning(f"⚠️ Error selecting initial feature: {e}")
                    # Fallback: select first feature
                    if feature_names:
                        selected_indices.append(0)
                        remaining_indices.remove(0)
                        mrmr_results['selected_features'].append(feature_names[0])
                        mrmr_results['mrmr_scores'][feature_names[0]] = 0.0

            # Iteratively select features
            while len(selected_indices) < n_features and remaining_indices:
                best_score = -np.inf
                best_idx = None

                for idx in remaining_indices:
                    feature_name = feature_names[idx]

                    # Calculate relevance
                    relevance = relevance_scores.get(feature_name, 0)

                    # Calculate redundancy with already selected features
                    redundancy = 0
                    if selected_indices:
                        redundancy_scores = []
                        for selected_idx in selected_indices:
                            selected_name = feature_names[selected_idx]
                            score = self._calculate_redundancy_score(
                                X[:, idx], X[:, selected_idx],
                                feature_name, selected_name, redundancy_method
                            )
                            redundancy_scores.append(score)
                        redundancy = np.mean(redundancy_scores)

                    # mRMR score
                    mrmr_score = relevance - redundancy

                    if mrmr_score > best_score:
                        best_score = mrmr_score
                        best_idx = idx

                if best_idx is not None:
                    selected_indices.append(best_idx)
                    remaining_indices.remove(best_idx)
                    feature_name = feature_names[best_idx]
                    mrmr_results['selected_features'].append(feature_name)
                    mrmr_results['mrmr_scores'][feature_name] = best_score

                    # Store individual scores
                    mrmr_results['feature_scores'][feature_name] = {
                        'relevance': relevance_scores.get(feature_name, 0),
                        'redundancy': redundancy,
                        'mrmr_score': best_score
                    }

            mrmr_results['selection_metadata']['n_features_selected'] = len(mrmr_results['selected_features'])

            self.logger.info(f"✅ mRMR selection completed: {len(mrmr_results['selected_features'])} features selected")
            return mrmr_results

        except Exception as e:
            self.logger.error(f"❌ mRMR selection failed: {e}")
            return {'error': str(e), 'selected_features': []}

    def stability_weighted_selection(self, X: np.ndarray, y: np.ndarray,
                                   feature_names: List[str],
                                   stability_scores: Dict[str, float],
                                   threshold: float = 0.6,
                                   n_features: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform stability-weighted feature selection.

        Args:
            X: Feature matrix
            y: Target array
            feature_names: List of feature names
            stability_scores: Dictionary of feature stability scores
            threshold: Minimum stability threshold
            n_features: Number of features to select (None for threshold-based)

        Returns:
            Dictionary with selected features and stability analysis
        """
        try:
            self.logger.info(f"🔍 Starting stability-weighted selection (threshold={threshold})")

            stability_results = {
                'selected_features': [],
                'stability_analysis': {},
                'selection_metadata': {
                    'method': 'stability_weighted',
                    'stability_threshold': threshold,
                    'n_features_requested': n_features
                }
            }

            # Calculate base importance scores
            importance_scores = self._calculate_importance_scores(X, y, feature_names)

            # Combine stability and importance
            combined_scores = {}
            for feature in feature_names:
                stability = stability_scores.get(feature, 0.5)
                importance = importance_scores.get(feature, 0.0)

                # Weighted combination
                combined_score = stability * importance

                combined_scores[feature] = {
                    'stability': stability,
                    'importance': importance,
                    'combined_score': combined_score,
                    'meets_threshold': stability >= threshold
                }

            stability_results['stability_analysis'] = combined_scores

            # Select features
            if n_features is None:
                # Threshold-based selection
                selected_features = [
                    feature for feature, scores in combined_scores.items()
                    if scores['meets_threshold']
                ]
            else:
                # Top-N selection
                sorted_features = sorted(
                    combined_scores.items(),
                    key=lambda x: x[1]['combined_score'],
                    reverse=True
                )
                selected_features = [feature for feature, _ in sorted_features[:n_features]]

            stability_results['selected_features'] = selected_features
            stability_results['selection_metadata']['n_features_selected'] = len(selected_features)

            # Stability statistics
            stabilities = [scores['stability'] for scores in combined_scores.values()]
            stability_results['selection_metadata']['stability_stats'] = {
                'mean_stability': np.mean(stabilities),
                'std_stability': np.std(stabilities),
                'min_stability': np.min(stabilities),
                'max_stability': np.max(stabilities),
                'stable_features': sum(1 for s in stabilities if s >= threshold)
            }

            self.logger.info(f"✅ Stability-weighted selection completed: "
                           f"{len(selected_features)} features selected")
            return stability_results

        except Exception as e:
            self.logger.error(f"❌ Stability-weighted selection failed: {e}")
            return {'error': str(e), 'selected_features': []}

    def correlation_based_filtering(self, X: np.ndarray, feature_names: List[str],
                                  correlation_threshold: float = 0.95,
                                  method: str = 'pearson') -> Dict[str, Any]:
        """
        Perform correlation-based feature filtering.

        Args:
            X: Feature matrix
            feature_names: List of feature names
            correlation_threshold: Correlation threshold for filtering
            method: Correlation method ('pearson', 'spearman')

        Returns:
            Dictionary with filtered features and correlation analysis
        """
        try:
            self.logger.info(f"🔍 Starting correlation-based filtering (threshold={correlation_threshold})")

            correlation_results = {
                'selected_features': feature_names.copy(),  # Start with all
                'removed_features': [],
                'correlation_matrix': {},
                'highly_correlated_pairs': [],
                'selection_metadata': {
                    'method': 'correlation_filtering',
                    'correlation_threshold': correlation_threshold,
                    'correlation_method': method
                }
            }

            # Calculate correlation matrix
            if method == 'pearson':
                corr_matrix = np.corrcoef(X.T)
            elif method == 'spearman':
                from scipy.stats import spearmanr
                corr_matrix = np.zeros((X.shape[1], X.shape[1]))
                for i in range(X.shape[1]):
                    for j in range(X.shape[1]):
                        if i != j:
                            corr, _ = spearmanr(X[:, i], X[:, j])
                            corr_matrix[i, j] = corr
                        else:
                            corr_matrix[i, j] = 1.0
            else:
                raise ValueError(f"Unsupported correlation method: {method}")

            # Store correlation matrix
            for i, feature_i in enumerate(feature_names):
                correlation_results['correlation_matrix'][feature_i] = {}
                for j, feature_j in enumerate(feature_names):
                    if i != j:
                        correlation_results['correlation_matrix'][feature_i][feature_j] = corr_matrix[i, j]

            # Find highly correlated pairs
            removed_features = set()
            for i in range(len(feature_names)):
                if feature_names[i] in removed_features:
                    continue

                for j in range(i + 1, len(feature_names)):
                    if feature_names[j] in removed_features:
                        continue

                    corr_value = abs(corr_matrix[i, j])
                    if corr_value >= correlation_threshold:
                        # Remove the feature with higher index (arbitrary choice)
                        removed_feature = feature_names[j]
                        removed_features.add(removed_feature)

                        correlation_results['highly_correlated_pairs'].append({
                            'feature1': feature_names[i],
                            'feature2': removed_feature,
                            'correlation': corr_value
                        })

            # Update selected features
            correlation_results['selected_features'] = [
                f for f in feature_names if f not in removed_features
            ]
            correlation_results['removed_features'] = list(removed_features)

            correlation_results['selection_metadata'].update({
                'n_features_original': len(feature_names),
                'n_features_selected': len(correlation_results['selected_features']),
                'n_features_removed': len(correlation_results['removed_features']),
                'n_correlated_pairs': len(correlation_results['highly_correlated_pairs'])
            })

            self.logger.info(f"✅ Correlation-based filtering completed: "
                           f"{len(correlation_results['selected_features'])} features retained, "
                           f"{len(correlation_results['removed_features'])} removed")
            return correlation_results

        except Exception as e:
            self.logger.error(f"❌ Correlation-based filtering failed: {e}")
            return {'error': str(e), 'selected_features': feature_names}

    def recursive_feature_elimination(self, model: Any, X: np.ndarray, y: np.ndarray,
                                    feature_names: List[str], n_features: int,
                                    cv: int = 3) -> Dict[str, Any]:
        """
        Perform recursive feature elimination.

        Args:
            model: Base model for RFE
            X: Feature matrix
            y: Target array
            feature_names: List of feature names
            n_features: Number of features to select
            cv: Number of cross-validation folds

        Returns:
            Dictionary with selected features and RFE results
        """
        try:
            self.logger.info(f"🔄 Starting recursive feature elimination for {n_features} features")

            rfe_results = {
                'selected_features': [],
                'feature_ranking': {},
                'feature_scores': {},
                'selection_metadata': {
                    'method': 'recursive_feature_elimination',
                    'n_features_requested': n_features,
                    'cv_folds': cv
                }
            }

            if not SKLEARN_AVAILABLE:
                raise ImportError("Scikit-learn required for recursive feature elimination")

            # Create RFE selector
            rfe_selector = RFE(
                estimator=model,
                n_features_to_select=n_features,
                step=self.method_configs['rfe']['step']
            )

            # Fit RFE
            rfe_selector.fit(X, y)

            # Get selected features
            selected_mask = rfe_selector.support_
            selected_indices = np.where(selected_mask)[0]

            rfe_results['selected_features'] = [
                feature_names[idx] for idx in selected_indices
            ]

            # Get feature ranking
            ranking = rfe_selector.ranking_
            for idx, feature_name in enumerate(feature_names):
                rfe_results['feature_ranking'][feature_name] = ranking[idx]

            # Calculate cross-validated scores for different feature subsets
            if cv > 1:
                rfecv = RFECV(
                    estimator=model,
                    step=self.method_configs['rfe']['step'],
                    cv=StratifiedKFold(cv),
                    scoring=self.method_configs['rfe']['scoring']
                )
                rfecv.fit(X, y)

                rfe_results['optimal_n_features'] = rfecv.n_features_
                rfe_results['cv_scores'] = rfecv.cv_results_['mean_test_score'].tolist()

            rfe_results['selection_metadata']['n_features_selected'] = len(rfe_results['selected_features'])

            self.logger.info(f"✅ Recursive feature elimination completed: "
                           f"{len(rfe_results['selected_features'])} features selected")
            return rfe_results

        except Exception as e:
            self.logger.error(f"❌ Recursive feature elimination failed: {e}")
            return {'error': str(e), 'selected_features': []}

    def feature_importance_ranking(self, model: Any, X: np.ndarray, y: np.ndarray,
                                 feature_names: List[str],
                                 method: str = 'permutation') -> Dict[str, Any]:
        """
        Rank features by importance using various methods.

        Args:
            model: Trained model
            X: Feature matrix
            y: Target array
            feature_names: List of feature names
            method: Importance calculation method ('permutation', 'tree_importance', 'coefficients')

        Returns:
            Dictionary with feature importance ranking
        """
        try:
            self.logger.info(f"📊 Calculating feature importance using {method} method")

            importance_results = {
                'feature_importance': {},
                'ranking': [],
                'selection_metadata': {
                    'method': 'feature_importance_ranking',
                    'importance_method': method
                }
            }

            if method == 'permutation':
                importance_scores = self._calculate_permutation_importance(model, X, y, feature_names)
            elif method == 'tree_importance':
                importance_scores = self._calculate_tree_importance(model, X, y, feature_names)
            elif method == 'coefficients':
                importance_scores = self._calculate_coefficient_importance(model, X, y, feature_names)
            elif method == 'shap':
                shap_result = self._calculate_shap_importance(model, X, feature_names)
                importance_scores = shap_result.get('importance_scores', {})
            else:
                raise ValueError(f"Unsupported importance method: {method}")

            importance_results['feature_importance'] = importance_scores

            # Create ranking
            sorted_features = sorted(
                importance_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )

            importance_results['ranking'] = [
                {'feature': feature, 'importance': score, 'rank': idx + 1}
                for idx, (feature, score) in enumerate(sorted_features)
            ]

            self.logger.info(f"✅ Feature importance ranking completed for {len(feature_names)} features")
            return importance_results

        except Exception as e:
            self.logger.error(f"❌ Feature importance ranking failed: {e}")
            return {'error': str(e), 'ranking': []}

    def _calculate_shap_importance(self, model: Any, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Calculate SHAP-based feature importance with robust fallbacks."""
        try:
            import shap
            self.logger.info("🧠 Computing SHAP values for interpretability")
            # Choose explainer based on model type
            explainer = None
            try:
                explainer = shap.Explainer(model, X)
            except Exception:
                try:
                    explainer = shap.KernelExplainer(lambda data: model.predict(data), X[: min(len(X), 200)])
                except Exception as e:
                    self.logger.warning(f"SHAP explainer creation failed: {e}")
                    return {'importance_scores': {}, 'error': str(e)}

            subset_size = min(1000, len(X))
            shap_values = explainer(X[:subset_size])

            import numpy as _np
            vals = getattr(shap_values, 'values', None)
            if vals is None:
                vals = _np.array(shap_values)

            if vals.ndim == 3:
                abs_mean = _np.mean(_np.mean(_np.abs(vals), axis=2), axis=0)
            else:
                abs_mean = _np.mean(_np.abs(vals), axis=0)

            scores = {name: float(abs_mean[i]) for i, name in enumerate(feature_names[: len(abs_mean)])}

            return {
                'importance_scores': scores,
                'method': 'shap',
                'n_samples_used': subset_size
            }
        except Exception as e:
            self.logger.warning(f"⚠️ SHAP importance failed: {e}")
            return {'importance_scores': {}, 'error': str(e)}

    def composite_feature_scoring(self, X: np.ndarray, y: np.ndarray,
                                feature_names: List[str],
                                methods: List[str] = None,
                                weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Calculate composite feature scores using multiple methods.

        Args:
            X: Feature matrix
            y: Target array
            feature_names: List of feature names
            methods: List of scoring methods to use
            weights: Weights for each method

        Returns:
            Dictionary with composite feature scores
        """
        try:
            if methods is None:
                methods = ['mutual_info', 'importance', 'stability']

            if weights is None:
                weights = {method: 1.0 / len(methods) for method in methods}

            self.logger.info(f"🔍 Calculating composite feature scores using {methods}")

            composite_results = {
                'composite_scores': {},
                'method_scores': {},
                'feature_ranking': [],
                'selection_metadata': {
                    'method': 'composite_scoring',
                    'methods_used': methods,
                    'method_weights': weights
                }
            }

            # Calculate scores for each method
            method_scores = {}
            for method in methods:
                if method == 'mutual_info':
                    scores = self._calculate_relevance_scores(X, y, feature_names, 'mutual_info')
                elif method == 'importance':
                    scores = self._calculate_importance_scores(X, y, feature_names)
                elif method == 'stability':
                    scores = self._calculate_stability_scores(X, feature_names)
                elif method == 'variance':
                    scores = self._calculate_variance_scores(X, feature_names)
                else:
                    self.logger.warning(f"Unknown scoring method: {method}")
                    continue

                method_scores[method] = scores

            composite_results['method_scores'] = method_scores

            # Calculate composite scores
            for feature in feature_names:
                composite_score = 0.0
                method_contributions = {}

                for method in methods:
                    if method in method_scores and feature in method_scores[method]:
                        score = method_scores[method][feature]
                        weight = weights.get(method, 1.0)
                        contribution = score * weight
                        composite_score += contribution
                        method_contributions[method] = contribution

                composite_results['composite_scores'][feature] = {
                    'composite_score': composite_score,
                    'method_contributions': method_contributions
                }

            # Create ranking
            sorted_features = sorted(
                composite_results['composite_scores'].items(),
                key=lambda x: x[1]['composite_score'],
                reverse=True
            )

            composite_results['feature_ranking'] = [
                {'feature': feature, 'composite_score': scores['composite_score'],
                 'rank': idx + 1, 'method_contributions': scores['method_contributions']}
                for idx, (feature, scores) in enumerate(sorted_features)
            ]

            self.logger.info(f"✅ Composite feature scoring completed for {len(feature_names)} features")
            return composite_results

        except Exception as e:
            self.logger.error(f"❌ Composite feature scoring failed: {e}")
            return {'error': str(e), 'composite_scores': {}}

    def cross_validated_feature_selection(self, X: np.ndarray, y: np.ndarray,
                                        feature_names: List[str],
                                        cv_folds: int = 5,
                                        selection_method: str = 'importance') -> Dict[str, Any]:
        """
        Perform cross-validated feature selection for stability assessment.

        Args:
            X: Feature matrix
            y: Target array
            feature_names: List of feature names
            cv_folds: Number of cross-validation folds
            selection_method: Feature selection method

        Returns:
            Dictionary with cross-validated feature selection results
        """
        try:
            self.logger.info(f"🔄 Starting cross-validated feature selection ({cv_folds} folds)")

            cv_results = {
                'fold_selections': [],
                'feature_stability': {},
                'consensus_features': [],
                'selection_metadata': {
                    'method': 'cross_validated_selection',
                    'cv_folds': cv_folds,
                    'selection_method': selection_method
                }
            }

            if not SKLEARN_AVAILABLE:
                raise ImportError("Scikit-learn required for cross-validated feature selection")

            # Perform cross-validation
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

            fold_selections = []
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                try:
                    X_fold, y_fold = X[train_idx], y[train_idx]

                    # Perform feature selection on this fold
                    fold_selection = self._select_features_single_fold(
                        X_fold, y_fold, feature_names, selection_method
                    )

                    fold_selections.append({
                        'fold_idx': fold_idx,
                        'selected_features': fold_selection,
                        'n_features': len(fold_selection)
                    })

                except Exception as fold_e:
                    self.logger.warning(f"⚠️ Fold {fold_idx} feature selection failed: {fold_e}")
                    continue

            cv_results['fold_selections'] = fold_selections

            # Calculate feature stability
            if fold_selections:
                cv_results['feature_stability'] = self._calculate_feature_stability(
                    fold_selections, feature_names
                )

                # Find consensus features (selected in most folds)
                feature_counts = {}
                for fold in fold_selections:
                    for feature in fold['selected_features']:
                        feature_counts[feature] = feature_counts.get(feature, 0) + 1

                consensus_threshold = cv_folds * 0.6  # Selected in 60% of folds
                cv_results['consensus_features'] = [
                    feature for feature, count in feature_counts.items()
                    if count >= consensus_threshold
                ]

                cv_results['selection_metadata'].update({
                    'total_folds_completed': len(fold_selections),
                    'consensus_threshold': consensus_threshold,
                    'n_consensus_features': len(cv_results['consensus_features'])
                })

            self.logger.info(f"✅ Cross-validated feature selection completed: "
                           f"{len(cv_results['consensus_features'])} consensus features found")
            return cv_results

        except Exception as e:
            self.logger.error(f"❌ Cross-validated feature selection failed: {e}")
            return {'error': str(e), 'consensus_features': []}

    def _calculate_relevance_scores(self, X: np.ndarray, y: np.ndarray,
                                  feature_names: List[str], method: str) -> Dict[str, float]:
        """Calculate relevance scores for features."""
        try:
            scores = {}

            if method == 'mutual_info':
                if SKLEARN_AVAILABLE:
                    mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
                    scores = dict(zip(feature_names, mi_scores))
                else:
                    # Fallback: use correlation for regression-like relevance
                    for idx, feature_name in enumerate(feature_names):
                        try:
                            corr_matrix = np.corrcoef(X[:, idx], y)
                            if corr_matrix.ndim == 2 and corr_matrix.shape == (2, 2):
                                corr_value = corr_matrix[0, 1]
                            else:
                                corr_value = float(corr_matrix) if np.isscalar(corr_matrix) else 0.0
                            scores[feature_name] = abs(float(corr_value))
                        except (ValueError, IndexError, TypeError):
                            scores[feature_name] = 0.0

            elif method == 'correlation':
                for idx, feature_name in enumerate(feature_names):
                    try:
                        corr_matrix = np.corrcoef(X[:, idx], y)
                        if corr_matrix.ndim == 2 and corr_matrix.shape == (2, 2):
                            corr_value = corr_matrix[0, 1]
                        else:
                            corr_value = float(corr_matrix) if np.isscalar(corr_matrix) else 0.0
                        scores[feature_name] = abs(float(corr_value))
                    except (ValueError, IndexError, TypeError):
                        scores[feature_name] = 0.0

            elif method == 'importance':
                importance_scores = self._calculate_importance_scores(X, y, feature_names)
                scores = importance_scores

            # Handle NaN values
            for feature in feature_names:
                if feature not in scores or np.isnan(scores[feature]):
                    scores[feature] = 0.0

            return scores

        except Exception as e:
            self.logger.warning(f"Relevance score calculation failed: {e}")
            return {feature: 0.0 for feature in feature_names}

    def _calculate_redundancy_score(self, feature1: np.ndarray, feature2: np.ndarray,
                                  name1: str, name2: str, method: str) -> float:
        """Calculate redundancy score between two features."""
        try:
            if method == 'correlation':
                corr_matrix = np.corrcoef(feature1, feature2)
                if corr_matrix.ndim == 2 and corr_matrix.shape == (2, 2):
                    return abs(float(corr_matrix[0, 1]))
                else:
                    return abs(float(corr_matrix)) if np.isscalar(corr_matrix) else 0.0
            elif method == 'mutual_info':
                if SKLEARN_AVAILABLE:
                    mi_score = mutual_info_regression(feature1.reshape(-1, 1), feature2)[0]
                    return float(mi_score)
                else:
                    corr_matrix = np.corrcoef(feature1, feature2)
                    if corr_matrix.ndim == 2 and corr_matrix.shape == (2, 2):
                        return abs(float(corr_matrix[0, 1]))
                    else:
                        return abs(float(corr_matrix)) if np.isscalar(corr_matrix) else 0.0
            else:
                return 0.0
        except (ValueError, IndexError, TypeError, np.linalg.LinAlgError):
            return 0.0

    def _calculate_importance_scores(self, X: np.ndarray, y: np.ndarray,
                                   feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature importance scores using Random Forest."""
        try:
            if not SKLEARN_AVAILABLE:
                return {feature: 1.0 / len(feature_names) for feature in feature_names}

            # Choose appropriate model based on target
            if len(np.unique(y)) <= 10:  # Classification
                model = RandomForestClassifier(
                    n_estimators=self.method_configs['importance']['n_estimators'],
                    max_depth=self.method_configs['importance']['max_depth'],
                    random_state=self.random_state
                )
            else:  # Regression
                model = RandomForestRegressor(
                    n_estimators=self.method_configs['importance']['n_estimators'],
                    max_depth=self.method_configs['importance']['max_depth'],
                    random_state=self.random_state
                )

            model.fit(X, y)
            importance_scores = dict(zip(feature_names, model.feature_importances_))

            return importance_scores

        except Exception as e:
            self.logger.warning(f"Importance score calculation failed: {e}")
            return {feature: 1.0 / len(feature_names) for feature in feature_names}

    def _calculate_stability_scores(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature stability scores based on variance and distribution."""
        try:
            stability_scores = {}

            for idx, feature_name in enumerate(feature_names):
                feature_values = X[:, idx]

                # Remove NaN values for calculation
                clean_values = feature_values[~np.isnan(feature_values)]

                if len(clean_values) > 0:
                    # Stability based on coefficient of variation
                    mean_val = np.mean(clean_values)
                    std_val = np.std(clean_values)

                    if mean_val != 0:
                        cv = abs(std_val / mean_val)
                        # Convert to stability score (lower CV = higher stability)
                        stability = 1.0 / (1.0 + cv)
                    else:
                        stability = 0.5  # Neutral stability for zero-mean features
                else:
                    stability = 0.0

                stability_scores[feature_name] = stability

            return stability_scores

        except Exception as e:
            self.logger.warning(f"Stability score calculation failed: {e}")
            return {feature: 0.5 for feature in feature_names}

    def _calculate_variance_scores(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature scores based on variance."""
        try:
            variance_scores = {}

            for idx, feature_name in enumerate(feature_names):
                feature_values = X[:, idx]
                clean_values = feature_values[~np.isnan(feature_values)]

                if len(clean_values) > 0:
                    variance = np.var(clean_values)
                    # Normalize variance to [0, 1] range (roughly)
                    normalized_variance = min(variance / (variance + 1.0), 1.0)
                    variance_scores[feature_name] = normalized_variance
                else:
                    variance_scores[feature_name] = 0.0

            return variance_scores

        except Exception as e:
            self.logger.warning(f"Variance score calculation failed: {e}")
            return {feature: 0.0 for feature in feature_names}

    def _calculate_permutation_importance(self, model: Any, X: np.ndarray, y: np.ndarray,
                                        feature_names: List[str]) -> Dict[str, float]:
        """Calculate permutation feature importance."""
        try:
            if not SKLEARN_AVAILABLE:
                return self._calculate_importance_scores(X, y, feature_names)

            from sklearn.inspection import permutation_importance

            # Get baseline score
            baseline_score = self._calculate_model_score(model, X, y)

            # Calculate permutation importance
            perm_importance = permutation_importance(
                model, X, y, n_repeats=5, random_state=self.random_state
            )

            importance_scores = dict(zip(feature_names, perm_importance.importances_mean))
            return importance_scores

        except Exception as e:
            self.logger.warning(f"Permutation importance calculation failed: {e}")
            return self._calculate_importance_scores(X, y, feature_names)

    def _calculate_tree_importance(self, model: Any, X: np.ndarray, y: np.ndarray,
                                 feature_names: List[str]) -> Dict[str, float]:
        """Calculate tree-based feature importance."""
        try:
            if hasattr(model, 'feature_importances_'):
                return dict(zip(feature_names, model.feature_importances_))
            else:
                # Fallback to training a random forest
                return self._calculate_importance_scores(X, y, feature_names)
        except Exception as e:
            self.logger.warning(f"Tree importance calculation failed: {e}")
            return {feature: 1.0 / len(feature_names) for feature in feature_names}

    def _calculate_coefficient_importance(self, model: Any, X: np.ndarray, y: np.ndarray,
                                        feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature importance based on model coefficients."""
        try:
            if hasattr(model, 'coef_'):
                coefficients = np.abs(model.coef_.flatten())
                return dict(zip(feature_names, coefficients))
            elif hasattr(model, 'feature_importances_'):
                return dict(zip(feature_names, model.feature_importances_))
            else:
                return {feature: 1.0 / len(feature_names) for feature in feature_names}
        except Exception as e:
            self.logger.warning(f"Coefficient importance calculation failed: {e}")
            return {feature: 1.0 / len(feature_names) for feature in feature_names}

    def _calculate_model_score(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate a baseline score for the model."""
        try:
            if hasattr(model, 'score'):
                return model.score(X, y)
            else:
                # Fallback to accuracy for classification
                predictions = model.predict(X)
                if len(np.unique(y)) <= 10:  # Classification
                    from sklearn.metrics import accuracy_score
                    return accuracy_score(y, predictions)
                else:  # Regression
                    from sklearn.metrics import r2_score
                    return r2_score(y, predictions)
        except:
            return 0.5

    def _select_features_single_fold(self, X: np.ndarray, y: np.ndarray,
                                   feature_names: List[str], method: str) -> List[str]:
        """Select features for a single CV fold."""
        try:
            if method == 'importance':
                importance_scores = self._calculate_importance_scores(X, y, feature_names)
                # Select top 50% of features
                n_select = max(1, len(feature_names) // 2)
                sorted_features = sorted(
                    importance_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                return [feature for feature, _ in sorted_features[:n_select]]
            else:
                # Default: return all features
                return feature_names
        except Exception as e:
            self.logger.warning(f"Single fold feature selection failed: {e}")
            return feature_names

    def _calculate_feature_stability(self, fold_selections: List[Dict[str, Any]],
                                   feature_names: List[str]) -> Dict[str, Any]:
        """Calculate feature selection stability across folds."""
        try:
            stability_scores = {}

            for feature in feature_names:
                selection_count = sum(
                    1 for fold in fold_selections
                    if feature in fold['selected_features']
                )

                stability = selection_count / len(fold_selections)
                stability_scores[feature] = {
                    'selection_frequency': selection_count,
                    'stability_score': stability,
                    'selected_in_folds': [fold['fold_idx'] for fold in fold_selections
                                        if feature in fold['selected_features']]
                }

            return stability_scores

        except Exception as e:
            self.logger.warning(f"Feature stability calculation failed: {e}")
            return {}
