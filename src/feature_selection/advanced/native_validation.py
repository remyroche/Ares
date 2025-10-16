"""
Native Validation Framework Integration

This module provides built-in cross-validation, stability metrics,
and consensus scoring for feature selection methods.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
from sklearn.model_selection import KFold, TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Import project utilities
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_performance, tprint_debug
from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager, HardwareConfig

logger = logging.getLogger(__name__)

class NativeValidationFramework:
    """Native validation framework with built-in CV and stability metrics."""

    def __init__(self, config, hardware_manager: Optional[UnifiedHardwareManager] = None):
        """Initialize native validation framework."""
        self.config = config
        self.hardware_manager = hardware_manager
        self.logger = logger.getChild('NativeValidationFramework')

        # Validation tracking
        self.validation_history = []
        self.stability_history = defaultdict(list)
        self.consensus_history = defaultdict(list)

        # Performance tracking
        self.performance_stats = {
            'total_validations': 0,
            'cv_completions': 0,
            'stability_checks': 0,
            'consensus_checks': 0,
            'avg_validation_time': 0.0
        }

        tprint_success("🔧 NativeValidationFramework initialized")

    def validate_selection_methods(self, X: np.ndarray, y: np.ndarray,
                                 method_results: Dict[str, Dict[str, Any]],
                                 feature_names: List[str]) -> Dict[str, Any]:
        """Validate multiple selection methods with built-in CV."""
        if not self.config.enable_native_validation:
            return self._create_default_validation_results(method_results)

        tprint_debug("🔧 Running native validation for selection methods")

        start_time = time.time()

        try:
            validation_results = {
                'timestamp': time.time(),
                'data_shape': X.shape,
                'n_methods': len(method_results)
            }

            # Cross-validation validation
            if self.config.enable_native_validation:
                cv_results = self._run_cross_validation_validation(X, y, method_results, feature_names)
                validation_results['cross_validation'] = cv_results
                self.performance_stats['cv_completions'] += 1

            # Stability metrics
            if self.config.enable_stability_metrics:
                stability_results = self._calculate_stability_metrics(X, y, method_results, feature_names)
                validation_results['stability_metrics'] = stability_results
                self.performance_stats['stability_checks'] += 1

            # Consensus scoring
            if self.config.enable_consensus_scoring:
                consensus_results = self._calculate_consensus_scores(method_results, feature_names)
                validation_results['consensus_scores'] = consensus_results
                self.performance_stats['consensus_checks'] += 1

            # Performance validation
            if self.config.enable_performance_validation:
                performance_results = self._validate_performance(X, y, method_results, feature_names)
                validation_results['performance_validation'] = performance_results

            # Overall validation summary
            validation_results['overall_success'] = self._evaluate_overall_success(validation_results)

            # Update statistics
            end_time = time.time()
            execution_time = end_time - start_time
            self.performance_stats['total_validations'] += 1
            self.performance_stats['avg_validation_time'] = (
                (self.performance_stats['avg_validation_time'] * (self.performance_stats['total_validations'] - 1) +
                 execution_time) / self.performance_stats['total_validations']
            )

            validation_results['execution_time'] = execution_time

            # Store validation history
            self.validation_history.append(validation_results)

            tprint_success(f"✅ Native validation completed in {execution_time:.3f}s")
            return validation_results

        except Exception as e:
            self.logger.error(f"Native validation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }

    def _run_cross_validation_validation(self, X: np.ndarray, y: np.ndarray,
                                       method_results: Dict[str, Dict[str, Any]],
                                       feature_names: List[str]) -> Dict[str, Any]:
        """Run cross-validation validation for all methods."""
        tprint_debug("🔧 Running cross-validation validation")

        try:
            # Create CV splits
            cv_splits = self._create_cv_splits(X, y)

            # Validate each method
            method_cv_results = {}
            for method_name, result in method_results.items():
                if result.get('success', False):
                    method_cv_results[method_name] = self._validate_method_cv(
                        X, y, result, feature_names, cv_splits
                    )

            # Aggregate results
            aggregated_results = self._aggregate_cv_results(method_cv_results)

            return {
                'success': True,
                'method_results': method_cv_results,
                'aggregated_results': aggregated_results,
                'n_folds': len(cv_splits)
            }

        except Exception as e:
            self.logger.error(f"CV validation failed: {e}")
            return {'success': False, 'error': str(e)}

    def _create_cv_splits(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create cross-validation splits."""
        if self.config.cv_strategy == 'timeseries':
            cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        elif self.config.cv_strategy == 'stratified':
            # Determine if classification
            is_classification = len(np.unique(y)) < 10 and np.all(y == y.astype(int))
            if is_classification:
                cv = StratifiedKFold(n_splits=self.config.cv_folds, random_state=42)
            else:
                cv = KFold(n_splits=self.config.cv_folds, random_state=42)
        else:  # kfold
            cv = KFold(n_splits=self.config.cv_folds, random_state=42)

        return list(cv.split(X, y))

    def _validate_method_cv(self, X: np.ndarray, y: np.ndarray,
                          method_result: Dict[str, Any],
                          feature_names: List[str],
                          cv_splits: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Validate a single method using cross-validation."""
        try:
            selected_features = method_result.get('selected_features', [])
            selected_indices = [feature_names.index(f) for f in selected_features if f in feature_names]

            if not selected_indices:
                return {'success': False, 'error': 'No valid selected features'}

            # Evaluate on each fold
            fold_results = []
            for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Get selected features for this fold
                X_selected = X_test[:, selected_indices]

                # Calculate performance metrics
                fold_metrics = self._calculate_fold_metrics(X_selected, y_test)
                fold_metrics['fold'] = fold_idx
                fold_results.append(fold_metrics)

            # Aggregate fold results
            aggregated_metrics = self._aggregate_fold_metrics(fold_results)

            return {
                'success': True,
                'fold_results': fold_results,
                'aggregated_metrics': aggregated_metrics,
                'selected_features': selected_features,
                'n_selected': len(selected_features)
            }

        except Exception as e:
            self.logger.error(f"Method CV validation failed: {e}")
            return {'success': False, 'error': str(e)}

    def _calculate_fold_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Calculate performance metrics for a single fold."""
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
                    'accuracy': float(accuracy),
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
                    'mse': float(mse),
                    'r2': float(r2),
                    'model_type': 'regression'
                }

        except Exception as e:
            self.logger.warning(f"Fold metrics calculation failed: {e}")
            return {'error': str(e)}

    def _aggregate_fold_metrics(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across folds."""
        if not fold_results:
            return {}

        # Get all metric names
        all_metrics = set()
        for result in fold_results:
            all_metrics.update(result.keys())

        # Calculate statistics for each metric
        aggregated = {}
        for metric in all_metrics:
            if metric in ['fold', 'error', 'model_type']:
                continue

            values = [result[metric] for result in fold_results if metric in result and isinstance(result[metric], (int, float))]

            if values:
                aggregated[f'{metric}_mean'] = float(np.mean(values))
                aggregated[f'{metric}_std'] = float(np.std(values))
                aggregated[f'{metric}_min'] = float(np.min(values))
                aggregated[f'{metric}_max'] = float(np.max(values))

        return aggregated

    def _calculate_stability_metrics(self, X: np.ndarray, y: np.ndarray,
                                   method_results: Dict[str, Dict[str, Any]],
                                   feature_names: List[str]) -> Dict[str, Any]:
        """Calculate stability metrics for feature selection."""
        tprint_debug("🔧 Calculating stability metrics")

        try:
            stability_results = {}

            for method_name, result in method_results.items():
                if result.get('success', False):
                    selected_features = result.get('selected_features', [])
                    stability_score = self._calculate_method_stability(
                        X, y, selected_features, feature_names
                    )
                    stability_results[method_name] = stability_score

            # Calculate overall stability
            overall_stability = self._calculate_overall_stability(stability_results)

            return {
                'success': True,
                'method_stability': stability_results,
                'overall_stability': overall_stability
            }

        except Exception as e:
            self.logger.error(f"Stability metrics calculation failed: {e}")
            return {'success': False, 'error': str(e)}

    def _calculate_method_stability(self, X: np.ndarray, y: np.ndarray,
                                  selected_features: List[str],
                                  feature_names: List[str]) -> Dict[str, Any]:
        """Calculate stability for a single method."""
        try:
            # Bootstrap stability
            n_bootstrap = self.config.stability_n_bootstrap
            stability_scores = []

            for i in range(n_bootstrap):
                # Bootstrap sample
                n_samples = X.shape[0]
                bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_bootstrap = X[bootstrap_indices]
                y_bootstrap = y[bootstrap_indices]

                # Calculate feature importance for bootstrap
                rf = RandomForestRegressor(n_estimators=50, random_state=42)
                rf.fit(X_bootstrap, y_bootstrap)
                importance = rf.feature_importances_

                # Check if selected features are in top features
                selected_indices = [feature_names.index(f) for f in selected_features if f in feature_names]
                if selected_indices:
                    top_features = np.argsort(importance)[-len(selected_indices):]
                    stability = len(set(selected_indices) & set(top_features)) / len(selected_indices)
                    stability_scores.append(stability)

            if stability_scores:
                return {
                    'stability_mean': float(np.mean(stability_scores)),
                    'stability_std': float(np.std(stability_scores)),
                    'stability_min': float(np.min(stability_scores)),
                    'stability_max': float(np.max(stability_scores)),
                    'is_stable': np.mean(stability_scores) >= self.config.stability_threshold
                }
            else:
                return {
                    'stability_mean': 0.0,
                    'stability_std': 0.0,
                    'stability_min': 0.0,
                    'stability_max': 0.0,
                    'is_stable': False
                }

        except Exception as e:
            self.logger.warning(f"Method stability calculation failed: {e}")
            return {'error': str(e)}

    def _calculate_overall_stability(self, method_stability: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall stability across methods."""
        if not method_stability:
            return {'overall_stability': 0.0, 'stable_methods': 0, 'total_methods': 0}

        stability_scores = []
        stable_methods = 0

        for method, stability in method_stability.items():
            if 'stability_mean' in stability:
                stability_scores.append(stability['stability_mean'])
                if stability.get('is_stable', False):
                    stable_methods += 1

        if stability_scores:
            overall_stability = np.mean(stability_scores)
        else:
            overall_stability = 0.0

        return {
            'overall_stability': float(overall_stability),
            'stable_methods': stable_methods,
            'total_methods': len(method_stability),
            'stability_ratio': stable_methods / len(method_stability) if method_stability else 0.0
        }

    def _calculate_consensus_scores(self, method_results: Dict[str, Dict[str, Any]],
                                  feature_names: List[str]) -> Dict[str, Any]:
        """Calculate consensus scores for feature selection."""
        tprint_debug("🔧 Calculating consensus scores")

        try:
            # Extract selected features from each method
            method_selections = {}
            for method_name, result in method_results.items():
                if result.get('success', False):
                    selected_features = result.get('selected_features', [])
                    method_selections[method_name] = set(selected_features)
                else:
                    method_selections[method_name] = set()

            # Calculate consensus for each feature
            feature_consensus = {}
            for feature in feature_names:
                selection_count = sum(1 for selections in method_selections.values() if feature in selections)
                total_methods = len(method_selections)

                consensus_score = selection_count / total_methods if total_methods > 0 else 0.0
                is_consensus = selection_count >= self.config.consensus_min_methods

                feature_consensus[feature] = {
                    'consensus_score': float(consensus_score),
                    'selection_count': selection_count,
                    'total_methods': total_methods,
                    'is_consensus': is_consensus
                }

            # Calculate overall consensus
            consensus_features = [f for f, c in feature_consensus.items() if c['is_consensus']]
            overall_consensus = len(consensus_features) / len(feature_names) if feature_names else 0.0

            return {
                'success': True,
                'feature_consensus': feature_consensus,
                'consensus_features': consensus_features,
                'n_consensus_features': len(consensus_features),
                'overall_consensus': float(overall_consensus)
            }

        except Exception as e:
            self.logger.error(f"Consensus scoring failed: {e}")
            return {'success': False, 'error': str(e)}

    def _validate_performance(self, X: np.ndarray, y: np.ndarray,
                            method_results: Dict[str, Dict[str, Any]],
                            feature_names: List[str]) -> Dict[str, Any]:
        """Validate performance of selected features."""
        tprint_debug("🔧 Validating performance")

        try:
            performance_results = {}

            for method_name, result in method_results.items():
                if result.get('success', False):
                    selected_features = result.get('selected_features', [])
                    performance_score = self._calculate_performance_score(
                        X, y, selected_features, feature_names
                    )
                    performance_results[method_name] = performance_score

            return {
                'success': True,
                'method_performance': performance_results
            }

        except Exception as e:
            self.logger.error(f"Performance validation failed: {e}")
            return {'success': False, 'error': str(e)}

    def _calculate_performance_score(self, X: np.ndarray, y: np.ndarray,
                                   selected_features: List[str],
                                   feature_names: List[str]) -> Dict[str, Any]:
        """Calculate performance score for selected features."""
        try:
            selected_indices = [feature_names.index(f) for f in selected_features if f in feature_names]

            if not selected_indices:
                return {'performance_score': 0.0, 'error': 'No valid selected features'}

            X_selected = X[:, selected_indices]

            # Determine if classification or regression
            is_classification = len(np.unique(y)) < 10 and np.all(y == y.astype(int))

            if is_classification:
                model = LogisticRegression(random_state=42, max_iter=1000)
                model.fit(X_selected, y)
                y_pred = model.predict(X_selected)
                performance_score = accuracy_score(y, y_pred)
            else:
                model = LinearRegression()
                model.fit(X_selected, y)
                y_pred = model.predict(X_selected)
                performance_score = r2_score(y, y_pred)

            return {
                'performance_score': float(performance_score),
                'model_type': 'classification' if is_classification else 'regression'
            }

        except Exception as e:
            self.logger.warning(f"Performance score calculation failed: {e}")
            return {'performance_score': 0.0, 'error': str(e)}

    def _evaluate_overall_success(self, validation_results: Dict[str, Any]) -> bool:
        """Evaluate overall validation success."""
        try:
            # Check CV success
            cv_success = validation_results.get('cross_validation', {}).get('success', False)

            # Check stability success
            stability_success = validation_results.get('stability_metrics', {}).get('success', False)

            # Check consensus success
            consensus_success = validation_results.get('consensus_scores', {}).get('success', False)

            # Check performance success
            performance_success = validation_results.get('performance_validation', {}).get('success', False)

            # Overall success requires at least CV and one other validation
            return cv_success and (stability_success or consensus_success or performance_success)

        except Exception as e:
            self.logger.warning(f"Overall success evaluation failed: {e}")
            return False

    def _create_default_validation_results(self, method_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Create default validation results when validation is disabled."""
        return {
            'success': True,
            'cross_validation': {'success': True, 'message': 'Validation disabled'},
            'stability_metrics': {'success': True, 'message': 'Validation disabled'},
            'consensus_scores': {'success': True, 'message': 'Validation disabled'},
            'performance_validation': {'success': True, 'message': 'Validation disabled'},
            'overall_success': True
        }

    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        stats = self.performance_stats.copy()

        # Add validation history insights
        if self.validation_history:
            stats['validation_history_size'] = len(self.validation_history)
            stats['recent_validations'] = len([v for v in self.validation_history if v.get('overall_success', False)])
            stats['success_rate'] = stats['recent_validations'] / stats['validation_history_size'] if stats['validation_history_size'] > 0 else 0.0

        return stats

    def get_validation_insights(self) -> Dict[str, Any]:
        """Get insights about validation performance."""
        insights = {
            'total_validations': self.performance_stats['total_validations'],
            'success_rate': 0.0,
            'avg_validation_time': self.performance_stats['avg_validation_time'],
            'validation_trends': {}
        }

        if self.validation_history:
            successful_validations = [v for v in self.validation_history if v.get('overall_success', False)]
            insights['success_rate'] = len(successful_validations) / len(self.validation_history)

            # Analyze trends
            if len(self.validation_history) > 1:
                recent_validations = self.validation_history[-5:]  # Last 5 validations
                recent_success_rate = sum(1 for v in recent_validations if v.get('overall_success', False)) / len(recent_validations)
                insights['validation_trends']['recent_success_rate'] = recent_success_rate
                insights['validation_trends']['trend'] = 'improving' if recent_success_rate > insights['success_rate'] else 'declining' if recent_success_rate < insights['success_rate'] else 'stable'

        return insights
