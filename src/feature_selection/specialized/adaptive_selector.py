"""
Adaptive Feature Selection for Small Sample SR Level Analysis

This module provides intelligent feature selection that adapts to the available sample size,
using various techniques to maximize learning while preventing overfitting.

Key Features:
- Adaptive sample size requirements based on available data
- Intelligent feature reduction using multiple selection methods
- Conservative learning with small samples
- Progressive feature selection (start simple, add complexity)
- Cross-validation with small sample handling
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression, 
    RFE, SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr, pearsonr
import warnings

# Import utilities
from src.utils.tprint import tprint, tprint_warning, tprint_error, tprint_success, tprint_debug
from src.utils.math_validation import validate_numeric_array, validate_finite, validate_positive

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class AdaptiveFeatureSelectionConfig:
    """Configuration for adaptive feature selection."""
    # Sample size thresholds
    min_samples_absolute: int = 10  # Absolute minimum samples
    min_samples_per_feature: float = 3.0  # Minimum samples per feature
    ideal_samples_per_feature: float = 10.0  # Ideal samples per feature
    
    # Feature selection methods
    use_variance_threshold: bool = True
    use_correlation_filter: bool = True
    use_mutual_information: bool = True
    use_recursive_elimination: bool = True
    use_lasso_selection: bool = True
    use_random_forest: bool = True
    
    # Selection parameters
    max_features_absolute: int = 20  # Maximum features regardless of sample size
    correlation_threshold: float = 0.95  # Remove highly correlated features
    variance_threshold: float = 0.01  # Remove low variance features
    mutual_info_k: int = 5  # Number of features to select with mutual info
    
    # Cross-validation parameters
    cv_folds: int = 3  # Use fewer folds for small samples
    test_size: float = 0.2
    
    # Conservative learning parameters
    conservative_mode_threshold: int = 50  # Use conservative mode below this sample size
    regularization_strength: float = 0.1  # Higher regularization for small samples
    
    # Progressive selection
    enable_progressive_selection: bool = True
    progressive_stages: int = 3  # Number of progressive selection stages

@dataclass
class FeatureSelectionResult:
    """Result of feature selection process."""
    selected_features: List[str]
    feature_scores: Dict[str, float]
    selection_method: str
    sample_size: int
    n_features_selected: int
    n_features_available: int
    overfitting_risk: str  # 'low', 'medium', 'high'
    selection_confidence: float
    method_details: Dict[str, Any] = field(default_factory=dict)

class AdaptiveFeatureSelector:
    """Adaptive feature selector that works with small samples."""
    
    def __init__(self, config: Optional[AdaptiveFeatureSelectionConfig] = None):
        self.config = config or AdaptiveFeatureSelectionConfig()
        self.logger = logger.getChild('AdaptiveFeatureSelector')
        
        self.logger.info("Initializing AdaptiveFeatureSelector")
        self.logger.info(f"Configuration: min_samples={self.config.min_samples_absolute}, "
                        f"min_samples_per_feature={self.config.min_samples_per_feature}")
    
    def select_features(self, X: pd.DataFrame, y: np.ndarray, 
                       feature_names: Optional[List[str]] = None) -> FeatureSelectionResult:
        """
        Select features adaptively based on sample size and data characteristics.
        
        Args:
            X: Feature matrix (samples x features)
            y: Target variable
            feature_names: Optional list of feature names
            
        Returns:
            FeatureSelectionResult with selected features and metadata
        """
        try:
            self.logger.info(f"Starting adaptive feature selection: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Validate inputs
            if X.shape[0] < self.config.min_samples_absolute:
                self.logger.warning(f"Insufficient samples: {X.shape[0]} < {self.config.min_samples_absolute}")
                return self._create_minimal_selection(X, y, feature_names)
            
            # Calculate adaptive parameters
            n_samples, n_features = X.shape
            max_features = self._calculate_max_features(n_samples, n_features)
            
            self.logger.info(f"Adaptive parameters: max_features={max_features}, "
                           f"overfitting_risk={self._assess_overfitting_risk(n_samples, n_features)}")
            
            # Progressive feature selection for small samples
            if self.config.enable_progressive_selection and n_samples < self.config.conservative_mode_threshold:
                return self._progressive_feature_selection(X, y, feature_names, max_features)
            
            # Standard feature selection
            return self._standard_feature_selection(X, y, feature_names, max_features)
            
        except Exception as e:
            self.logger.error(f"Feature selection failed: {e}")
            return self._create_fallback_selection(X, y, feature_names)
    
    def _calculate_max_features(self, n_samples: int, n_features: int) -> int:
        """Calculate maximum features based on sample size."""
        # Conservative approach: use the more restrictive limit
        by_samples = int(n_samples / self.config.min_samples_per_feature)
        by_absolute = self.config.max_features_absolute
        
        max_features = min(by_samples, by_absolute, n_features)
        
        self.logger.info(f"Max features calculation: by_samples={by_samples}, "
                        f"by_absolute={by_absolute}, final={max_features}")
        
        return max_features
    
    def _assess_overfitting_risk(self, n_samples: int, n_features: int) -> str:
        """Assess overfitting risk based on sample size and feature count."""
        ratio = n_samples / n_features
        
        if ratio >= self.config.ideal_samples_per_feature:
            return 'low'
        elif ratio >= self.config.min_samples_per_feature:
            return 'medium'
        else:
            return 'high'
    
    def _progressive_feature_selection(self, X: pd.DataFrame, y: np.ndarray, 
                                     feature_names: Optional[List[str]], 
                                     max_features: int) -> FeatureSelectionResult:
        """Progressive feature selection for small samples."""
        self.logger.info("Using progressive feature selection for small samples")
        
        current_features = list(X.columns) if feature_names is None else feature_names
        current_X = X.copy()
        
        # Stage 1: Remove low variance and highly correlated features
        self.logger.info("Stage 1: Removing low variance and highly correlated features")
        current_X, current_features = self._remove_low_variance_features(current_X, current_features)
        current_X, current_features = self._remove_correlated_features(current_X, current_features)
        
        # Stage 2: Use simple statistical methods
        if len(current_features) > max_features:
            self.logger.info("Stage 2: Using statistical feature selection")
            current_X, current_features = self._statistical_feature_selection(
                current_X, y, current_features, max_features
            )
        
        # Stage 3: Use model-based selection if still too many features
        if len(current_features) > max_features:
            self.logger.info("Stage 3: Using model-based feature selection")
            current_X, current_features = self._model_based_selection(
                current_X, y, current_features, max_features
            )
        
        # Calculate final metrics
        n_samples = X.shape[0]
        overfitting_risk = self._assess_overfitting_risk(n_samples, len(current_features))
        confidence = self._calculate_selection_confidence(n_samples, len(current_features))
        
        return FeatureSelectionResult(
            selected_features=current_features,
            feature_scores={},  # Will be filled by individual methods
            selection_method='progressive',
            sample_size=n_samples,
            n_features_selected=len(current_features),
            n_features_available=X.shape[1],
            overfitting_risk=overfitting_risk,
            selection_confidence=confidence,
            method_details={'stages_completed': 3}
        )
    
    def _standard_feature_selection(self, X: pd.DataFrame, y: np.ndarray, 
                                  feature_names: Optional[List[str]], 
                                  max_features: int) -> FeatureSelectionResult:
        """Standard feature selection for larger samples."""
        self.logger.info("Using standard feature selection")
        
        current_features = list(X.columns) if feature_names is None else feature_names
        current_X = X.copy()
        
        # Remove low variance features
        if self.config.use_variance_threshold:
            current_X, current_features = self._remove_low_variance_features(current_X, current_features)
        
        # Remove highly correlated features
        if self.config.use_correlation_filter:
            current_X, current_features = self._remove_correlated_features(current_X, current_features)
        
        # Use mutual information if still too many features
        if len(current_features) > max_features and self.config.use_mutual_information:
            current_X, current_features = self._mutual_information_selection(
                current_X, y, current_features, max_features
            )
        
        # Use recursive elimination if still too many features
        if len(current_features) > max_features and self.config.use_recursive_elimination:
            current_X, current_features = self._recursive_elimination_selection(
                current_X, y, current_features, max_features
            )
        
        # Use Lasso selection if still too many features
        if len(current_features) > max_features and self.config.use_lasso_selection:
            current_X, current_features = self._lasso_selection(
                current_X, y, current_features, max_features
            )
        
        # Calculate final metrics
        n_samples = X.shape[0]
        overfitting_risk = self._assess_overfitting_risk(n_samples, len(current_features))
        confidence = self._calculate_selection_confidence(n_samples, len(current_features))
        
        return FeatureSelectionResult(
            selected_features=current_features,
            feature_scores={},
            selection_method='standard',
            sample_size=n_samples,
            n_features_selected=len(current_features),
            n_features_available=X.shape[1],
            overfitting_risk=overfitting_risk,
            selection_confidence=confidence
        )
    
    def _remove_low_variance_features(self, X: pd.DataFrame, feature_names: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Remove features with low variance."""
        try:
            selector = VarianceThreshold(threshold=self.config.variance_threshold)
            X_selected = selector.fit_transform(X)
            
            # Get selected feature names
            selected_indices = selector.get_support(indices=True)
            selected_features = [feature_names[i] for i in selected_indices]
            
            self.logger.info(f"Variance threshold: removed {len(feature_names) - len(selected_features)} features")
            
            return pd.DataFrame(X_selected, columns=selected_features), selected_features
            
        except Exception as e:
            self.logger.warning(f"Variance threshold failed: {e}")
            return X, feature_names
    
    def _remove_correlated_features(self, X: pd.DataFrame, feature_names: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Remove highly correlated features."""
        try:
            # Calculate correlation matrix
            corr_matrix = X.corr().abs()
            
            # Find pairs of highly correlated features
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find features to drop
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.config.correlation_threshold)]
            
            # Keep features
            selected_features = [f for f in feature_names if f not in to_drop]
            X_selected = X[selected_features]
            
            self.logger.info(f"Correlation filter: removed {len(to_drop)} highly correlated features")
            
            return X_selected, selected_features
            
        except Exception as e:
            self.logger.warning(f"Correlation filter failed: {e}")
            return X, feature_names
    
    def _statistical_feature_selection(self, X: pd.DataFrame, y: np.ndarray, 
                                     feature_names: List[str], max_features: int) -> Tuple[pd.DataFrame, List[str]]:
        """Use statistical methods for feature selection."""
        try:
            # Use F-test for feature selection
            selector = SelectKBest(score_func=f_regression, k=min(max_features, len(feature_names)))
            X_selected = selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_indices = selector.get_support(indices=True)
            selected_features = [feature_names[i] for i in selected_indices]
            
            self.logger.info(f"Statistical selection: selected {len(selected_features)} features using F-test")
            
            return pd.DataFrame(X_selected, columns=selected_features), selected_features
            
        except Exception as e:
            self.logger.warning(f"Statistical selection failed: {e}")
            return X, feature_names
    
    def _mutual_information_selection(self, X: pd.DataFrame, y: np.ndarray, 
                                    feature_names: List[str], max_features: int) -> Tuple[pd.DataFrame, List[str]]:
        """Use mutual information for feature selection."""
        try:
            # Calculate mutual information scores
            mi_scores = mutual_info_regression(X, y, random_state=42)
            
            # Select top features
            k = min(max_features, len(feature_names))
            top_indices = np.argsort(mi_scores)[-k:]
            
            selected_features = [feature_names[i] for i in top_indices]
            X_selected = X[selected_features]
            
            self.logger.info(f"Mutual information: selected {len(selected_features)} features")
            
            return X_selected, selected_features
            
        except Exception as e:
            self.logger.warning(f"Mutual information selection failed: {e}")
            return X, feature_names
    
    def _recursive_elimination_selection(self, X: pd.DataFrame, y: np.ndarray, 
                                       feature_names: List[str], max_features: int) -> Tuple[pd.DataFrame, List[str]]:
        """Use recursive feature elimination."""
        try:
            # Preprocess data to handle infinity and large values
            X_processed = X.copy()

            # Handle infinity values
            inf_mask = np.isinf(X_processed.values)
            if np.any(inf_mask):
                self.logger.warning(f"⚠️ Found {np.sum(inf_mask)} infinity values in data for adaptive RFE, replacing with finite values")

                # Replace positive infinity
                pos_inf_mask = np.isposinf(X_processed.values)
                if np.any(pos_inf_mask):
                    finite_mask = np.isfinite(X_processed.values)
                    if np.any(finite_mask):
                        max_finite = np.max(X_processed.values[finite_mask])
                        X_processed.values[pos_inf_mask] = max(max_finite * 10, 1e10)
                    else:
                        X_processed.values[pos_inf_mask] = 1e10

                # Replace negative infinity
                neg_inf_mask = np.isneginf(X_processed.values)
                if np.any(neg_inf_mask):
                    finite_mask = np.isfinite(X_processed.values)
                    if np.any(finite_mask):
                        min_finite = np.min(X_processed.values[finite_mask])
                        X_processed.values[neg_inf_mask] = min(min_finite * 10, -1e10)
                    else:
                        X_processed.values[neg_inf_mask] = -1e10

            # Clip extremely large values
            max_float64 = 1e308
            min_float64 = -1e308
            X_processed = X_processed.clip(min_float64, max_float64)

            # Use processed data for RFE
            X = X_processed

            # Use Random Forest for RFE
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            selector = RFE(estimator, n_features_to_select=max_features)
            X_selected = selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_indices = selector.get_support(indices=True)
            selected_features = [feature_names[i] for i in selected_indices]
            
            self.logger.info(f"Recursive elimination: selected {len(selected_features)} features")
            
            return pd.DataFrame(X_selected, columns=selected_features), selected_features
            
        except Exception as e:
            self.logger.warning(f"Recursive elimination failed: {e}")
            return X, feature_names
    
    def _lasso_selection(self, X: pd.DataFrame, y: np.ndarray, 
                        feature_names: List[str], max_features: int) -> Tuple[pd.DataFrame, List[str]]:
        """Use Lasso for feature selection."""
        try:
            # Use Lasso with cross-validation
            lasso = LassoCV(cv=min(3, X.shape[0] - 1), random_state=42, 
                           alpha=self.config.regularization_strength)
            lasso.fit(X, y)
            
            # Get non-zero coefficients
            selected_indices = np.where(lasso.coef_ != 0)[0]
            selected_features = [feature_names[i] for i in selected_indices]
            
            # If still too many features, select top by absolute coefficient value
            if len(selected_features) > max_features:
                coef_abs = np.abs(lasso.coef_[selected_indices])
                top_indices = np.argsort(coef_abs)[-max_features:]
                selected_features = [selected_features[i] for i in top_indices]
            
            X_selected = X[selected_features]
            
            self.logger.info(f"Lasso selection: selected {len(selected_features)} features")
            
            return X_selected, selected_features
            
        except Exception as e:
            self.logger.warning(f"Lasso selection failed: {e}")
            return X, feature_names
    
    def _model_based_selection(self, X: pd.DataFrame, y: np.ndarray, 
                             feature_names: List[str], max_features: int) -> Tuple[pd.DataFrame, List[str]]:
        """Use model-based feature selection."""
        try:
            # Use Random Forest feature importance
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X, y)
            
            # Get feature importance
            importance_scores = rf.feature_importances_
            
            # Select top features
            k = min(max_features, len(feature_names))
            top_indices = np.argsort(importance_scores)[-k:]
            
            selected_features = [feature_names[i] for i in top_indices]
            X_selected = X[selected_features]
            
            self.logger.info(f"Model-based selection: selected {len(selected_features)} features")
            
            return X_selected, selected_features
            
        except Exception as e:
            self.logger.warning(f"Model-based selection failed: {e}")
            return X, feature_names
    
    def _calculate_selection_confidence(self, n_samples: int, n_features: int) -> float:
        """Calculate confidence in feature selection based on sample size."""
        ratio = n_samples / n_features
        
        if ratio >= self.config.ideal_samples_per_feature:
            return 0.9
        elif ratio >= self.config.min_samples_per_feature:
            return 0.7
        else:
            return 0.5
    
    def _create_minimal_selection(self, X: pd.DataFrame, y: np.ndarray, 
                                feature_names: Optional[List[str]]) -> FeatureSelectionResult:
        """Create minimal feature selection for very small samples."""
        self.logger.warning("Creating minimal feature selection for very small samples")
        
        if feature_names is None:
            feature_names = list(X.columns)
        
        # Select only the most important features (top 3-5)
        n_select = min(3, len(feature_names))
        
        # Use simple correlation with target
        try:
            correlations = []
            for col in X.columns:
                corr, _ = pearsonr(X[col], y)
                correlations.append(abs(corr))
            
            top_indices = np.argsort(correlations)[-n_select:]
            selected_features = [feature_names[i] for i in top_indices]
            
        except Exception as minimal_e:
            tprint_warning(f"⚠️ Minimal selection failed: {minimal_e}")
            # Fallback: select first few features
            selected_features = feature_names[:n_select]
        
        return FeatureSelectionResult(
            selected_features=selected_features,
            feature_scores={},
            selection_method='minimal',
            sample_size=X.shape[0],
            n_features_selected=len(selected_features),
            n_features_available=len(feature_names),
            overfitting_risk='high',
            selection_confidence=0.3,
            method_details={'fallback_reason': 'insufficient_samples'}
        )
    
    def _create_fallback_selection(self, X: pd.DataFrame, y: np.ndarray, 
                                 feature_names: Optional[List[str]]) -> FeatureSelectionResult:
        """Create fallback selection when all methods fail."""
        self.logger.warning("Creating fallback feature selection")
        
        if feature_names is None:
            feature_names = list(X.columns)
        
        # Select first few features as fallback
        n_select = min(5, len(feature_names))
        selected_features = feature_names[:n_select]
        
        return FeatureSelectionResult(
            selected_features=selected_features,
            feature_scores={},
            selection_method='fallback',
            sample_size=X.shape[0],
            n_features_selected=len(selected_features),
            n_features_available=len(feature_names),
            overfitting_risk='high',
            selection_confidence=0.2,
            method_details={'fallback_reason': 'selection_failed'}
        )

def get_adaptive_feature_selector(config: Optional[AdaptiveFeatureSelectionConfig] = None) -> AdaptiveFeatureSelector:
    """Get an adaptive feature selector instance."""
    return AdaptiveFeatureSelector(config)