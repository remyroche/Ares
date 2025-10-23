"""
ML Common - Feature Selection Backward Compatibility Module

This module provides backward compatibility for feature selection functionality.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class FeatureSelectionConfig:
    """Configuration for feature selection operations."""
    method: str = "recursive"
    max_features: Optional[int] = None
    min_features: int = 1
    cv_folds: int = 5
    scoring: str = "neg_mean_squared_error"
    random_state: int = 42
    n_jobs: int = -1

class FeatureSelector:
    """Backward-compatible feature selector wrapper."""

    def __init__(self, config: Optional[FeatureSelectionConfig] = None):
        self.config = config or FeatureSelectionConfig()
        self.selected_features: List[str] = []
        self.feature_scores: Dict[str, float] = {}
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureSelector':
        """Fit the feature selector to the data."""
        try:
            # Simple feature selection based on correlation with target
            if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
                correlations = {}
                for col in X.columns:
                    if X[col].dtype in ['int64', 'float64']:
                        corr = abs(X[col].corr(y))
                        if not np.isnan(corr):
                            correlations[col] = corr

                # Sort by correlation and select top features
                sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
                max_features = self.config.max_features or len(sorted_features)

                self.selected_features = [feat for feat, score in sorted_features[:max_features]]
                self.feature_scores = dict(sorted_features[:max_features])
                self._is_fitted = True

            return self

        except Exception as e:
            # Fallback: select all features
            self.selected_features = list(X.columns) if hasattr(X, 'columns') else []
            self.feature_scores = {}
            self._is_fitted = True
            return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data by selecting features."""
        if not self._is_fitted:
            raise ValueError("FeatureSelector must be fitted before transform")

        if not self.selected_features:
            return X

        # Ensure selected features exist in the data
        available_features = [feat for feat in self.selected_features if feat in X.columns]
        if not available_features:
            # Fallback: return first few columns
            available_features = list(X.columns)[:min(10, len(X.columns))]

        return X[available_features]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit the selector and transform the data."""
        return self.fit(X, y).transform(X)

    def get_support(self) -> List[bool]:
        """Get boolean mask of selected features."""
        if not hasattr(self, '_feature_names_in'):
            return []
        return [feat in self.selected_features for feat in self._feature_names_in]

    def get_feature_names_out(self) -> List[str]:
        """Get names of selected features."""
        return self.selected_features.copy()

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self.feature_scores.copy()

    @property
    def n_features_in_(self) -> int:
        """Number of features seen during fit."""
        return len(getattr(self, '_feature_names_in', []))

    @property
    def n_features_out_(self) -> int:
        """Number of features selected."""
        return len(self.selected_features)

# Convenience functions for backward compatibility
def create_feature_selector(config: Optional[FeatureSelectionConfig] = None) -> FeatureSelector:
    """Create a feature selector instance."""
    return FeatureSelector(config)

def select_features(X: pd.DataFrame, y: pd.Series,
                   method: str = "correlation",
                   max_features: Optional[int] = None) -> List[str]:
    """Select features using the specified method."""
    config = FeatureSelectionConfig(method=method, max_features=max_features)
    selector = FeatureSelector(config)
    selector.fit(X, y)
    return selector.selected_features

__all__ = [
    'FeatureSelector',
    'FeatureSelectionConfig',
    'create_feature_selector',
    'select_features'
]
