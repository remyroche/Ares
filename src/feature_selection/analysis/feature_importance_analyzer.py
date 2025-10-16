"""
Feature Importance Analyzer

This module provides comprehensive feature importance analysis capabilities
for machine learning models, supporting multiple importance calculation methods.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
from datetime import datetime
import time

# Import required libraries with fallbacks
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LassoCV, Lasso
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.feature_selection import f_classif, f_regression
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Some feature importance methods will be disabled.")

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Some statistical methods will be disabled.")

# Import project utilities
from src.utils.tprint import tprint
from src.utils.logger import system_logger

# Configure logging
_LOGGER = system_logger.getChild('FeatureImportanceAnalyzer')

class ImportanceMethod(Enum):
    """Available feature importance calculation methods."""
    RANDOM_FOREST = "random_forest"
    LASSO = "lasso"
    MUTUAL_INFO = "mutual_info"
    F_SCORE = "f_score"
    CORRELATION = "correlation"
    PERMUTATION = "permutation"
    VARIANCE = "variance"
    CHI2 = "chi2"

@dataclass
class FeatureImportanceConfig:
    """Configuration for feature importance analysis."""
    methods: List[ImportanceMethod] = field(default_factory=lambda: [
        ImportanceMethod.RANDOM_FOREST,
        ImportanceMethod.LASSO,
        ImportanceMethod.MUTUAL_INFO
    ])
    top_k_features: Optional[int] = None
    enable_parallel: bool = True
    n_jobs: int = -1
    random_state: int = 42
    temporal_window: Optional[int] = None
    min_importance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    verbose: bool = True

@dataclass
class FeatureImportanceResult:
    """Result of feature importance analysis."""
    feature_names: List[str]
    importance_scores: Dict[str, Dict[str, float]]
    method_weights: Dict[str, float]
    top_features: List[str]
    analysis_metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

    def get_combined_scores(self) -> Dict[str, float]:
        """Get combined importance scores across all methods."""
        combined = {}
        for feature in self.feature_names:
            scores = []
            weights = []
            for method, scores_dict in self.importance_scores.items():
                if feature in scores_dict:
                    scores.append(scores_dict[feature])
                    weights.append(self.method_weights.get(method, 1.0))

            if scores:
                # Weighted average
                combined[feature] = np.average(scores, weights=weights)
            else:
                combined[feature] = 0.0

        return combined

class FeatureImportanceAnalyzer:
    """Comprehensive feature importance analyzer."""

    def __init__(self, config: Optional[FeatureImportanceConfig] = None):
        """Initialize the analyzer with configuration."""
        self.config = config or FeatureImportanceConfig()
        self.logger = _LOGGER
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None

        # Validate configuration
        self._validate_config()

        tprint(f"🔍 FeatureImportanceAnalyzer initialized with methods: {[m.value for m in self.config.methods]}")

    def _validate_config(self) -> None:
        """Validate the configuration."""
        if not self.config.methods:
            raise ValueError("At least one importance method must be specified")

        if not SKLEARN_AVAILABLE and any(method in [
            ImportanceMethod.RANDOM_FOREST, ImportanceMethod.LASSO,
            ImportanceMethod.MUTUAL_INFO, ImportanceMethod.F_SCORE,
            ImportanceMethod.PERMUTATION
        ] for method in self.config.methods):
            raise ImportError("scikit-learn is required for the specified methods")

        if not SCIPY_AVAILABLE and ImportanceMethod.CORRELATION in self.config.methods:
            raise ImportError("scipy is required for correlation analysis")

    def analyze_features(self, X: Union[np.ndarray, pd.DataFrame],
                        y: Union[np.ndarray, pd.Series],
                        feature_names: Optional[List[str]] = None,
                        task_type: str = 'auto') -> FeatureImportanceResult:
        """
        Analyze feature importance using specified methods.

        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Names of features
            task_type: 'classification', 'regression', or 'auto'

        Returns:
            FeatureImportanceResult with analysis results
        """
        start_time = time.time()

        # Prepare data
        X_processed, y_processed, feature_names_processed = self._prepare_data(X, y, feature_names)

        # Determine task type
        if task_type == 'auto':
            task_type = self._detect_task_type(y_processed)

        tprint(f"📊 Analyzing {len(feature_names_processed)} features using {len(self.config.methods)} methods")

        # Calculate importance scores
        importance_scores = {}
        method_weights = {}

        for method in self.config.methods:
            try:
                tprint(f"🔍 Calculating importance using {method.value}")
                scores = self._calculate_importance(
                    X_processed, y_processed, method, task_type
                )
                importance_scores[method.value] = scores
                method_weights[method.value] = 1.0  # Equal weights for now

            except Exception as e:
                self.logger.warning(f"Failed to calculate {method.value} importance: {e}")
                continue

        if not importance_scores:
            raise RuntimeError("All importance calculation methods failed")

        # Create result
        result = FeatureImportanceResult(
            feature_names=feature_names_processed,
            importance_scores=importance_scores,
            method_weights=method_weights,
            top_features=self._get_top_features(importance_scores, method_weights),
            analysis_metadata={
                'task_type': task_type,
                'n_features': len(feature_names_processed),
                'n_samples': len(X_processed),
                'methods_used': list(importance_scores.keys()),
                'analysis_time': time.time() - start_time,
                'config': self.config.__dict__
            }
        )

        tprint(f"✅ Feature importance analysis completed in {result.analysis_metadata['analysis_time']:.2f}s")
        return result

    def _prepare_data(self, X: Union[np.ndarray, pd.DataFrame],
                     y: Union[np.ndarray, pd.Series],
                     feature_names: Optional[List[str]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare and validate input data."""
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            if feature_names is None:
                feature_names = list(X.columns)
        else:
            X_array = np.asarray(X)
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]

        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = np.asarray(y)

        # Handle missing values
        if np.isnan(X_array).any():
            self.logger.warning("⚠️ Missing values detected in features, filling with median")
            X_array = np.nan_to_num(X_array, nan=np.nanmedian(X_array))

        if np.isnan(y_array).any():
            self.logger.warning("⚠️ Missing values detected in target, removing samples")
            valid_mask = ~np.isnan(y_array)
            X_array = X_array[valid_mask]
            y_array = y_array[valid_mask]

        return X_array, y_array, feature_names

    def _detect_task_type(self, y: np.ndarray) -> str:
        """Detect if this is a classification or regression task."""
        unique_values = len(np.unique(y))
        if unique_values <= 10 and np.all(y == y.astype(int)):
            return 'classification'
        else:
            return 'regression'

    def _calculate_importance(self, X: np.ndarray, y: np.ndarray,
                            method: ImportanceMethod, task_type: str) -> Dict[str, float]:
        """Calculate importance scores for a specific method."""
        if method == ImportanceMethod.RANDOM_FOREST:
            return self._random_forest_importance(X, y, task_type)
        elif method == ImportanceMethod.LASSO:
            return self._lasso_importance(X, y)
        elif method == ImportanceMethod.MUTUAL_INFO:
            return self._mutual_info_importance(X, y, task_type)
        elif method == ImportanceMethod.F_SCORE:
            return self._f_score_importance(X, y, task_type)
        elif method == ImportanceMethod.CORRELATION:
            return self._correlation_importance(X, y)
        elif method == ImportanceMethod.PERMUTATION:
            return self._permutation_importance(X, y, task_type)
        elif method == ImportanceMethod.VARIANCE:
            return self._variance_importance(X)
        else:
            raise ValueError(f"Unknown importance method: {method}")

    def _random_forest_importance(self, X: np.ndarray, y: np.ndarray, task_type: str) -> Dict[str, float]:
        """Calculate Random Forest feature importance."""
        if task_type == 'classification':
            model = RandomForestClassifier(
                n_estimators=100, random_state=self.config.random_state, n_jobs=self.config.n_jobs
            )
        else:
            model = RandomForestRegressor(
                n_estimators=100, random_state=self.config.random_state, n_jobs=self.config.n_jobs
            )

        model.fit(X, y)
        importance_scores = model.feature_importances_

        return {f"feature_{i}": float(score) for i, score in enumerate(importance_scores)}

    def _lasso_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate Lasso feature importance."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X) if self.scaler else X

        # Use LassoCV for automatic alpha selection
        lasso = LassoCV(cv=5, random_state=self.config.random_state, n_jobs=self.config.n_jobs)
        lasso.fit(X_scaled, y)

        # Use absolute coefficients as importance
        importance_scores = np.abs(lasso.coef_)

        return {f"feature_{i}": float(score) for i, score in enumerate(importance_scores)}

    def _mutual_info_importance(self, X: np.ndarray, y: np.ndarray, task_type: str) -> Dict[str, float]:
        """Calculate mutual information feature importance."""
        if task_type == 'classification':
            scores = mutual_info_classif(X, y, random_state=self.config.random_state)
        else:
            scores = mutual_info_regression(X, y, random_state=self.config.random_state)

        return {f"feature_{i}": float(score) for i, score in enumerate(scores)}

    def _f_score_importance(self, X: np.ndarray, y: np.ndarray, task_type: str) -> Dict[str, float]:
        """Calculate F-score feature importance."""
        if task_type == 'classification':
            scores, _ = f_classif(X, y)
        else:
            scores, _ = f_regression(X, y)

        # Normalize scores
        scores = scores / np.sum(scores) if np.sum(scores) > 0 else scores

        return {f"feature_{i}": float(score) for i, score in enumerate(scores)}

    def _correlation_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate correlation-based feature importance."""
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for correlation analysis")

        scores = []
        for i in range(X.shape[1]):
            if np.std(X[:, i]) > 0:  # Avoid division by zero
                corr, _ = stats.pearsonr(X[:, i], y)
                scores.append(abs(corr))
            else:
                scores.append(0.0)

        return {f"feature_{i}": float(score) for i, score in enumerate(scores)}

    def _permutation_importance(self, X: np.ndarray, y: np.ndarray, task_type: str) -> Dict[str, float]:
        """Calculate permutation feature importance."""
        if task_type == 'classification':
            model = RandomForestClassifier(
                n_estimators=50, random_state=self.config.random_state, n_jobs=self.config.n_jobs
            )
        else:
            model = RandomForestRegressor(
                n_estimators=50, random_state=self.config.random_state, n_jobs=self.config.n_jobs
            )

        model.fit(X, y)

        # Calculate permutation importance
        perm_importance = permutation_importance(
            model, X, y, n_repeats=5, random_state=self.config.random_state, n_jobs=self.config.n_jobs
        )

        return {f"feature_{i}": float(score) for i, score in enumerate(perm_importance.importances_mean)}

    def _variance_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Calculate variance-based feature importance."""
        scores = np.var(X, axis=0)
        # Normalize
        scores = scores / np.sum(scores) if np.sum(scores) > 0 else scores

        return {f"feature_{i}": float(score) for i, score in enumerate(scores)}

    def _get_top_features(self, importance_scores: Dict[str, Dict[str, float]],
                         method_weights: Dict[str, float]) -> List[str]:
        """Get top features based on combined importance scores."""
        # Calculate combined scores
        combined_scores = {}
        for feature in list(importance_scores.values())[0].keys():
            scores = []
            weights = []
            for method, scores_dict in importance_scores.items():
                if feature in scores_dict:
                    scores.append(scores_dict[feature])
                    weights.append(method_weights.get(method, 1.0))

            if scores:
                combined_scores[feature] = np.average(scores, weights=weights)

        # Sort by importance
        sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        # Return top k features
        if self.config.top_k_features:
            return [feature for feature, _ in sorted_features[:self.config.top_k_features]]
        else:
            return [feature for feature, _ in sorted_features]

# Convenience functions
def analyze_feature_importance(X: Union[np.ndarray, pd.DataFrame],
                             y: Union[np.ndarray, pd.Series],
                             methods: Optional[List[ImportanceMethod]] = None,
                             top_k: Optional[int] = None,
                             **kwargs) -> FeatureImportanceResult:
    """
    Convenience function for quick feature importance analysis.

    Args:
        X: Feature matrix
        y: Target vector
        methods: List of importance methods to use
        top_k: Number of top features to return
        **kwargs: Additional configuration parameters

    Returns:
        FeatureImportanceResult with analysis results
    """
    config = FeatureImportanceConfig(
        methods=methods or [ImportanceMethod.RANDOM_FOREST, ImportanceMethod.LASSO],
        top_k_features=top_k,
        **kwargs
    )

    analyzer = FeatureImportanceAnalyzer(config)
    return analyzer.analyze_features(X, y)

def get_important_features(X: Union[np.ndarray, pd.DataFrame],
                         y: Union[np.ndarray, pd.Series],
                         k: int = 10,
                         method: ImportanceMethod = ImportanceMethod.RANDOM_FOREST,
                         **kwargs) -> List[str]:
    """
    Get top k important features using a single method.

    Args:
        X: Feature matrix
        y: Target vector
        k: Number of top features to return
        method: Importance method to use
        **kwargs: Additional configuration parameters

    Returns:
        List of top k feature names
    """
    config = FeatureImportanceConfig(
        methods=[method],
        top_k_features=k,
        **kwargs
    )

    analyzer = FeatureImportanceAnalyzer(config)
    result = analyzer.analyze_features(X, y)
    return result.top_features

# Export all public classes and functions
__all__ = [
    'FeatureImportanceAnalyzer',
    'FeatureImportanceConfig',
    'FeatureImportanceResult',
    'ImportanceMethod',
    'analyze_feature_importance',
    'get_important_features'
]
