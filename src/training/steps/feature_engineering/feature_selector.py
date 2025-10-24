"""
Tactician Feature Selector

This module provides feature selection utilities for tactician models with
comprehensive type safety and error handling.
"""

import logging
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

# Import ML libraries with fallback support
try:
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from src.utils.tprint import tprint_info, tprint_success, tprint_error, tprint_warning

# Import our custom types
from ..step_types import (
    StepConfig, FeatureSelectionResult, ValidationResult, MetricsDict,
    DataFrameType, SeriesType, SignalType,
    FeatureSelectionError, ValidationError, ConfigurationError,
    validate_config, create_error_result, create_success_result,
    is_dataframe, is_series
)

logger = logging.getLogger(__name__)

class TacticianFeatureSelector:
    """
    Feature selector for tactician models with multiple selection strategies.
    
    Provides comprehensive feature selection with type safety and error handling.
    """

    def __init__(self, config: Optional[StepConfig] = None):
        """
        Initialize the feature selector.

        Args:
            config: Configuration dictionary

        Raises:
            ConfigurationError: If configuration is invalid
            ImportError: If required ML libraries are not available
        """
        try:
            # Validate configuration
            self.config = validate_config(config) if config else {}
            
            # Initialize feature selection parameters
            self.selected_features: List[str] = []
            self.feature_scores: Dict[str, float] = {}
            self.selection_method = self.config.get('selection_method', 'mutual_info')
            self.n_features = self.config.get('n_features', 50)
            
            # Validate selection method
            valid_methods = ['mutual_info', 'f_classif', 'random_forest']
            if self.selection_method not in valid_methods:
                raise ConfigurationError(f"Invalid selection_method: {self.selection_method}. Must be one of {valid_methods}")
            
            # Validate number of features
            if not isinstance(self.n_features, int) or self.n_features <= 0:
                raise ConfigurationError(f"n_features must be a positive integer, got: {self.n_features}")
            
            # Check ML library availability
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn is required for feature selection but not available")
            
            tprint_success(f"✅ TacticianFeatureSelector initialized with method: {self.selection_method}")
            
        except Exception as e:
            tprint_error(f"❌ Failed to initialize TacticianFeatureSelector: {e}")
            raise ConfigurationError(f"Feature selector initialization failed: {e}") from e

    async def select_features(
        self,
        X: DataFrameType,
        y: SeriesType,
        signal_type: SignalType = 'long'
    ) -> FeatureSelectionResult:
        """
        Select features for the given signal type with comprehensive error handling.

        Args:
            X: Feature matrix
            y: Target labels
            signal_type: Type of signal ('long' or 'short')

        Returns:
            FeatureSelectionResult containing selection results

        Raises:
            FeatureSelectionError: If feature selection fails
            ValidationError: If input data is invalid
        """
        try:
            # Validate input data
            if not is_dataframe(X):
                raise ValidationError("X must be a pandas DataFrame")
            if not is_series(y):
                raise ValidationError("y must be a pandas Series")
            if signal_type not in ['long', 'short']:
                raise ValidationError(f"Invalid signal_type: {signal_type}. Must be 'long' or 'short'")
            
            tprint_info(f"🔍 Selecting features for {signal_type} signals using {self.selection_method} method...")

            # Check for empty data
            if X.empty or y.empty:
                raise ValidationError("Input data cannot be empty")
            
            # Check data consistency
            if len(X) != len(y):
                raise ValidationError(f"X and y must have the same length: X={len(X)}, y={len(y)}")

            # Handle missing values
            X_clean = X.fillna(X.median())
            y_clean = y.fillna(0)

            # Validate cleaned data
            if X_clean.empty or y_clean.empty:
                raise ValidationError("Data became empty after cleaning missing values")

            # Select features based on method
            try:
                if self.selection_method == 'mutual_info':
                    selector = SelectKBest(
                        score_func=mutual_info_classif,
                        k=min(self.n_features, X_clean.shape[1])
                    )
                elif self.selection_method == 'f_classif':
                    selector = SelectKBest(
                        score_func=f_classif,
                        k=min(self.n_features, X_clean.shape[1])
                    )
                elif self.selection_method == 'random_forest':
                    # Use Random Forest for feature importance
                    rf = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf.fit(X_clean, y_clean)
                    feature_importance = rf.feature_importances_
                    
                    # Select top k features
                    feature_importance_df = pd.DataFrame({
                        'feature': X_clean.columns,
                        'importance': feature_importance
                    }).sort_values('importance', ascending=False)
                    
                    selected_features = feature_importance_df.head(self.n_features)['feature'].tolist()
                    self.selected_features = selected_features
                    self.feature_scores = dict(zip(selected_features, feature_importance_df.head(self.n_features)['importance']))
                    
                    tprint_success(f"✅ Selected {len(selected_features)} features for {signal_type} using Random Forest")
                    
                    return create_success_result(
                        selected_features=selected_features,
                        feature_scores=self.feature_scores,
                        n_features=len(selected_features),
                        signal_type=signal_type
                    )
                else:
                    raise FeatureSelectionError(f"Unknown selection method: {self.selection_method}")

                # Fit selector for sklearn methods
                X_selected = selector.fit_transform(X_clean, y_clean)
                selected_features = X_clean.columns[selector.get_support()].tolist()

                # Store results
                self.selected_features = selected_features
                self.feature_scores = dict(zip(selected_features, selector.scores_[selector.get_support()]))

                tprint_success(f"✅ Selected {len(selected_features)} features for {signal_type} using {self.selection_method}")

                return create_success_result(
                    selected_features=selected_features,
                    feature_scores=self.feature_scores,
                    n_features=len(selected_features),
                    signal_type=signal_type
                )

            except Exception as e:
                raise FeatureSelectionError(f"Feature selection algorithm failed: {e}") from e

        except ValidationError:
            raise
        except FeatureSelectionError:
            raise
        except Exception as e:
            tprint_error(f"❌ Unexpected error in feature selection: {e}")
            raise FeatureSelectionError(f"Feature selection failed: {e}") from e

    def get_feature_importance(self, X: DataFrameType, y: SeriesType) -> Dict[str, float]:
        """
        Get feature importance using Random Forest with comprehensive error handling.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            Dictionary with feature importance scores

        Raises:
            ValidationError: If input data is invalid
            FeatureSelectionError: If importance calculation fails
        """
        try:
            # Validate input data
            if not is_dataframe(X):
                raise ValidationError("X must be a pandas DataFrame")
            if not is_series(y):
                raise ValidationError("y must be a pandas Series")
            
            if X.empty or y.empty:
                raise ValidationError("Input data cannot be empty")
            
            if len(X) != len(y):
                raise ValidationError(f"X and y must have the same length: X={len(X)}, y={len(y)}")

            tprint_info("🔍 Calculating feature importance using Random Forest...")

            # Check ML library availability
            if not SKLEARN_AVAILABLE:
                raise FeatureSelectionError("scikit-learn is required for feature importance calculation but not available")

            # Clean data
            X_clean = X.fillna(X.median())
            y_clean = y.fillna(0)

            # Train Random Forest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_clean, y_clean)

            # Get importance scores
            importance_scores = dict(zip(X.columns, rf.feature_importances_))
            
            tprint_success(f"✅ Calculated feature importance for {len(importance_scores)} features")
            return importance_scores

        except ValidationError:
            raise
        except Exception as e:
            tprint_error(f"❌ Error calculating feature importance: {e}")
            raise FeatureSelectionError(f"Feature importance calculation failed: {e}") from e
